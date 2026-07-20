#!/usr/bin/env bash
set -euo pipefail

if ! command -v sandbox-exec >/dev/null 2>&1; then
  echo "probe-runtime-worker-sandbox: sandbox-exec not found" >&2
  exit 1
fi

if ! command -v clang >/dev/null 2>&1; then
  echo "probe-runtime-worker-sandbox: clang not found" >&2
  exit 1
fi

root="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/mlxfast-sandbox-probe.XXXXXX")"
root="$(cd -P "${root}" && pwd)"

probe_source="${root}/probe.c"
probe_bin="${root}/probe"
preference_probe_source="${root}/preference-probe.m"
preference_probe_bin="${root}/preference-probe"
preference_suite="org.mlxfast.sandbox-probe.$$.${RANDOM}"
profile_path="${root}/worker.sb"
private_dir="${root}/private"
golden_path="${private_dir}/correctness_golden.json"
private_path="${private_dir}/gpqa_reference_cases.json"
outside_write_path="${root}/outside-write.txt"
private_write_path="${private_dir}/private-write.txt"
unix_socket_path="${root}/probe.sock"
server_pid=""
sysv_server_pid=""
sysv_ready_path="${root}/sysv-ready"
sysv_key="$((0x4d000000 | ($$ & 0x00ffffff)))"

cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${sysv_server_pid}" ]]; then
    kill "${sysv_server_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -x "${preference_probe_bin}" ]]; then
    sandbox-exec -p '(version 1) (allow default)' \
      "${preference_probe_bin}" cleanup "${preference_suite}" \
      >/dev/null 2>&1 || true
    "${preference_probe_bin}" cleanup "${preference_suite}" \
      >/dev/null 2>&1 || true
  fi
  rm -rf "${root}"
}

mkdir -p "${private_dir}"
printf '%s\n' '{"secret":true}' > "${golden_path}"
printf '%s\n' '{"secret":true}' > "${private_path}"
trap cleanup EXIT

cat > "${probe_source}" <<'C'
#include <errno.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <netinet/in.h>
#include <os/log.h>
#include <semaphore.h>
#include <signal.h>
#include <spawn.h>
#include <servers/bootstrap.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/msg.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/un.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static int server_shm_id = -1;
static int server_sem_id = -1;
static int server_msg_id = -1;

static int is_denied_errno(int value) {
  return value == EACCES || value == EPERM;
}

static void fail_errno(const char *label) {
  fprintf(stderr, "%s: unexpected errno=%d (%s)\n", label, errno, strerror(errno));
  exit(1);
}

static void expect_read_denied(const char *label, const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd >= 0) {
    close(fd);
    fprintf(stderr, "%s: read unexpectedly succeeded\n", label);
    exit(1);
  }
  if (!is_denied_errno(errno)) {
    fail_errno(label);
  }
}

static void expect_write_denied(const char *label, const char *path) {
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd >= 0) {
    close(fd);
    fprintf(stderr, "%s: write unexpectedly succeeded\n", label);
    exit(1);
  }
  if (!is_denied_errno(errno)) {
    fail_errno(label);
  }
}

static void expect_dev_null_write_allowed(void) {
  int fd = open("/dev/null", O_WRONLY);
  if (fd < 0) {
    fail_errno("/dev/null write");
  }
  if (write(fd, "x", 1) != 1) {
    fail_errno("/dev/null write payload");
  }
  close(fd);
}

static int mach_lookup_succeeds(const char *name) {
  mach_port_t service = MACH_PORT_NULL;
  kern_return_t result = bootstrap_look_up(
    bootstrap_port,
    name,
    &service
  );
  if (result != KERN_SUCCESS) {
    return 0;
  }
  if (service != MACH_PORT_NULL) {
    (void)mach_port_deallocate(mach_task_self(), service);
  }
  return 1;
}

static void expect_pasteboard_lookup_denied(void) {
  if (mach_lookup_succeeds("com.apple.pasteboard.1")) {
    fprintf(stderr, "pasteboard Mach lookup unexpectedly succeeded\n");
    exit(1);
  }
}

static int logging_services_are_available(void) {
  return mach_lookup_succeeds("com.apple.logd")
    && mach_lookup_succeeds("com.apple.system.logger");
}

static void expect_logging_lookup_denied(void) {
  if (mach_lookup_succeeds("com.apple.logd")
      || mach_lookup_succeeds("com.apple.system.logger")) {
    fprintf(stderr, "unified logging Mach lookup unexpectedly succeeded\n");
    exit(1);
  }
  // Exercise the public API too: logging must degrade to a no-op rather than
  // changing stdout/stderr or crashing the worker when logd is unavailable.
  os_log(OS_LOG_DEFAULT, "mlxfast runtime worker sandbox log probe");
}

static void expect_posix_shm_denied(void) {
  char name[64];
  snprintf(name, sizeof(name), "/mlxfast-sandbox-shm-%ld", (long)getpid());
  int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd >= 0) {
    close(fd);
    (void)shm_unlink(name);
    fprintf(stderr, "POSIX named shared memory unexpectedly succeeded\n");
    exit(1);
  }
  if (!is_denied_errno(errno)) {
    fail_errno("POSIX named shared memory");
  }
}

static void expect_posix_semaphore_denied(void) {
  char name[64];
  snprintf(name, sizeof(name), "/mlxfast-sandbox-sem-%ld", (long)getpid());
  sem_t *value = sem_open(name, O_CREAT | O_EXCL, 0600, 1);
  if (value != SEM_FAILED) {
    (void)sem_close(value);
    (void)sem_unlink(name);
    fprintf(stderr, "POSIX named semaphore unexpectedly succeeded\n");
    exit(1);
  }
  if (!is_denied_errno(errno)) {
    fail_errno("POSIX named semaphore");
  }
}

// On tested macOS, Seatbelt does not mediate creation of a brand-new
// IPC_PRIVATE shared-memory ID through ipc-sysv-shm. Cross-process reuse still
// requires looking up or operating on an existing object, so the trusted
// unsandboxed server creates keyed objects and this sandboxed client proves
// those lookups are denied for shm, semaphores, and message queues.
static void expect_sysv_ipc_denied(const char *raw_key) {
  key_t key = (key_t)strtol(raw_key, NULL, 10);
  int shm_id = shmget(key, 4096, 0);
  if (shm_id >= 0) {
    fprintf(stderr, "System V shared-memory lookup unexpectedly succeeded\n");
    exit(1);
  }
  if (!is_denied_errno(errno)) {
    fail_errno("System V shared-memory lookup");
  }

  int sem_id = semget(key, 1, 0);
  if (sem_id >= 0) {
    fprintf(stderr, "System V semaphore lookup unexpectedly succeeded\n");
    exit(1);
  }
  if (!is_denied_errno(errno)) {
    fail_errno("System V semaphore lookup");
  }

  int msg_id = msgget(key, 0);
  if (msg_id >= 0) {
    fprintf(stderr, "System V message-queue lookup unexpectedly succeeded\n");
    exit(1);
  }
  if (!is_denied_errno(errno)) {
    fail_errno("System V message-queue lookup");
  }
}

static void cleanup_sysv_server(int signal_number) {
  (void)signal_number;
  if (server_shm_id >= 0) {
    (void)shmctl(server_shm_id, IPC_RMID, NULL);
  }
  if (server_sem_id >= 0) {
    (void)semctl(server_sem_id, 0, IPC_RMID);
  }
  if (server_msg_id >= 0) {
    (void)msgctl(server_msg_id, IPC_RMID, NULL);
  }
  _exit(0);
}

static int run_sysv_server(const char *raw_key) {
  key_t key = (key_t)strtol(raw_key, NULL, 10);
  server_shm_id = shmget(key, 4096, IPC_CREAT | IPC_EXCL | 0600);
  server_sem_id = semget(key, 1, IPC_CREAT | IPC_EXCL | 0600);
  server_msg_id = msgget(key, IPC_CREAT | IPC_EXCL | 0600);
  if (server_shm_id < 0 || server_sem_id < 0 || server_msg_id < 0) {
    cleanup_sysv_server(0);
    return 2;
  }
  signal(SIGTERM, cleanup_sysv_server);
  signal(SIGINT, cleanup_sysv_server);
  printf("ready\n");
  fflush(stdout);
  for (;;) {
    pause();
  }
}

static void expect_inet_network_denied(void) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    if (is_denied_errno(errno)) {
      return;
    }
    fail_errno("inet socket");
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(9);
  int result = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
  int saved_errno = errno;
  close(fd);
  if (result < 0 && is_denied_errno(saved_errno)) {
    return;
  }

  fprintf(stderr, "inet network unexpectedly reached socket/connect path errno=%d (%s)\n", saved_errno, strerror(saved_errno));
  exit(1);
}

static void expect_unix_network_denied(const char *path) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    if (is_denied_errno(errno)) {
      return;
    }
    fail_errno("unix socket");
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  size_t length = strlen(path);
  if (length >= sizeof(addr.sun_path)) {
    fprintf(stderr, "unix socket path too long\n");
    exit(1);
  }
  memcpy(addr.sun_path, path, length + 1);
  int result = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
  int saved_errno = errno;
  close(fd);
  if (result < 0 && is_denied_errno(saved_errno)) {
    return;
  }

  fprintf(stderr, "unix socket connect unexpectedly reached path errno=%d (%s)\n", saved_errno, strerror(saved_errno));
  exit(1);
}

static void expect_fork_denied(void) {
  pid_t pid = fork();
  if (pid < 0) {
    if (is_denied_errno(errno)) {
      return;
    }
    fail_errno("fork");
  }
  if (pid == 0) {
    _exit(33);
  }
  int status = 0;
  (void)waitpid(pid, &status, 0);
  fprintf(stderr, "fork unexpectedly succeeded\n");
  exit(1);
}

static void expect_spawn_denied(void) {
  pid_t pid = 0;
  char *argv[] = {"/bin/echo", "sandbox-probe", NULL};
  int result = posix_spawn(&pid, "/bin/echo", NULL, NULL, argv, environ);
  if (result != 0) {
    if (is_denied_errno(result)) {
      return;
    }
    fprintf(stderr, "posix_spawn: unexpected errno=%d (%s)\n", result, strerror(result));
    exit(1);
  }
  int status = 0;
  (void)waitpid(pid, &status, 0);
  fprintf(stderr, "posix_spawn unexpectedly succeeded\n");
  exit(1);
}

static int run_server(const char *path) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    fail_errno("server socket");
  }
  (void)unlink(path);
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  size_t length = strlen(path);
  if (length >= sizeof(addr.sun_path)) {
    fprintf(stderr, "server unix socket path too long\n");
    return 2;
  }
  memcpy(addr.sun_path, path, length + 1);
  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    fail_errno("server bind");
  }
  if (listen(fd, 1) < 0) {
    fail_errno("server listen");
  }
  for (;;) {
    pause();
  }
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--pasteboard-available") == 0) {
    if (!mach_lookup_succeeds("com.apple.pasteboard.1")) {
      fprintf(stderr, "pasteboard Mach service is unavailable before sandbox\n");
      return 1;
    }
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "--logging-available") == 0) {
    if (!logging_services_are_available()) {
      fprintf(stderr, "unified logging Mach services are unavailable before sandbox\n");
      return 1;
    }
    return 0;
  }
  if (argc == 3 && strcmp(argv[1], "--server") == 0) {
    return run_server(argv[2]);
  }
  if (argc == 3 && strcmp(argv[1], "--sysv-server") == 0) {
    return run_sysv_server(argv[2]);
  }
  if (argc != 7) {
    fprintf(stderr, "usage: %s golden private outside-write private-write unix-socket sysv-key\n", argv[0]);
    return 2;
  }
  expect_read_denied("golden file", argv[1]);
  expect_read_denied("private file", argv[2]);
  expect_write_denied("outside write", argv[3]);
  expect_write_denied("private write", argv[4]);
  expect_dev_null_write_allowed();
  expect_pasteboard_lookup_denied();
  expect_logging_lookup_denied();
  expect_posix_shm_denied();
  expect_posix_semaphore_denied();
  expect_sysv_ipc_denied(argv[6]);
  expect_inet_network_denied();
  expect_unix_network_denied(argv[5]);
  expect_fork_denied();
  expect_spawn_denied();
  printf("runtime worker sandbox probe passed\n");
  return 0;
}
C

cat > "${preference_probe_source}" <<'OBJC'
#import <Foundation/Foundation.h>
#include <string.h>

static CFStringRef string_from_argument(const char *value) {
  return CFStringCreateWithCString(
    kCFAllocatorDefault,
    value,
    kCFStringEncodingUTF8
  );
}

static int preference_equals(
  CFStringRef suite,
  CFStringRef key,
  CFStringRef expected
) {
  CFPropertyListRef value = CFPreferencesCopyAppValue(key, suite);
  if (value == NULL) {
    return 0;
  }
  int equal = CFEqual(value, expected);
  CFRelease(value);
  return equal;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 3) {
      fprintf(stderr, "usage: preference-probe seed|attempt|verify|cleanup suite\n");
      return 2;
    }
    CFStringRef suite = string_from_argument(argv[2]);
    if (suite == NULL) {
      return 2;
    }
    int result = 0;
    if (strcmp(argv[1], "seed") == 0) {
      CFPreferencesSetAppValue(CFSTR("readable"), CFSTR("baseline"), suite);
      CFPreferencesSetAppValue(CFSTR("marker"), NULL, suite);
      result = CFPreferencesAppSynchronize(suite) ? 0 : 1;
    } else if (strcmp(argv[1], "attempt") == 0) {
      if (!preference_equals(suite, CFSTR("readable"), CFSTR("baseline"))) {
        fprintf(stderr, "sandboxed preference read unexpectedly failed\n");
        result = 1;
      } else {
        CFPreferencesSetAppValue(
          CFSTR("marker"),
          CFSTR("prompt-derived-marker"),
          suite
        );
        (void)CFPreferencesAppSynchronize(suite);
      }
    } else if (strcmp(argv[1], "verify") == 0) {
      if (!preference_equals(suite, CFSTR("readable"), CFSTR("baseline"))) {
        fprintf(stderr, "fresh sandboxed preference read unexpectedly failed\n");
        result = 1;
      }
      CFPropertyListRef marker = CFPreferencesCopyAppValue(
        CFSTR("marker"),
        suite
      );
      if (marker != NULL) {
        fprintf(stderr, "fresh worker recovered persisted preference marker\n");
        CFRelease(marker);
        result = 1;
      }
    } else if (strcmp(argv[1], "cleanup") == 0) {
      CFPreferencesSetAppValue(CFSTR("readable"), NULL, suite);
      CFPreferencesSetAppValue(CFSTR("marker"), NULL, suite);
      (void)CFPreferencesAppSynchronize(suite);
    } else {
      result = 2;
    }
    CFRelease(suite);
    return result;
  }
}
OBJC

clang "${probe_source}" -o "${probe_bin}"
clang -fobjc-arc -framework Foundation \
  "${preference_probe_source}" -o "${preference_probe_bin}"
"${probe_bin}" --pasteboard-available
"${probe_bin}" --logging-available
sandbox-exec -p '(version 1) (allow default)' \
  "${preference_probe_bin}" cleanup "${preference_suite}"
sandbox-exec -p '(version 1) (allow default)' \
  "${preference_probe_bin}" seed "${preference_suite}"
sandbox-exec -p '(version 1) (allow default)' \
  "${preference_probe_bin}" verify "${preference_suite}"
sandbox-exec -p '(version 1) (allow default) (deny user-preference-write)' \
  "${preference_probe_bin}" verify "${preference_suite}"

"${probe_bin}" --server "${unix_socket_path}" &
server_pid="$!"
"${probe_bin}" --sysv-server "${sysv_key}" > "${sysv_ready_path}" &
sysv_server_pid="$!"
for _ in {1..50}; do
  if [[ -S "${unix_socket_path}" ]]; then
    break
  fi
  sleep 0.1
done
if [[ ! -S "${unix_socket_path}" ]]; then
  echo "probe-runtime-worker-sandbox: unix socket listener did not start" >&2
  exit 1
fi
for _ in {1..50}; do
  if [[ -s "${sysv_ready_path}" ]]; then
    break
  fi
  sleep 0.1
done
if [[ ! -s "${sysv_ready_path}" ]]; then
  echo "probe-runtime-worker-sandbox: System V IPC listener did not start" >&2
  exit 1
fi

cat > "${profile_path}" <<EOF
(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny process-exec*)
(allow process-exec (literal "${probe_bin}"))
(allow process-exec (literal "${preference_probe_bin}"))
(deny ipc-posix-shm*)
(deny ipc-posix-sem*)
(deny ipc-sysv*)
(allow ipc-posix-shm-read*
  (ipc-posix-name "apple.shm.notification_center")
  (ipc-posix-name "apple.shm.cfprefsd.daemon")
  (ipc-posix-name-prefix "apple.cfprefs.")
  (ipc-posix-name-prefix "apple.shm.cfprefsd."))
(deny user-preference-write)
(deny mach-lookup (global-name-prefix "com.apple.pasteboard."))
(deny mach-lookup (global-name-prefix "com.apple.logd"))
(deny mach-lookup (global-name "com.apple.system.logger"))
(deny file-write*)
(allow file-write* (literal "/dev/null"))
(deny file-read* (literal "${golden_path}"))
(deny file-read* (subpath "${private_dir}"))
(deny file-write* (subpath "${private_dir}"))
EOF

sandbox-exec -f "${profile_path}" \
  "${preference_probe_bin}" attempt "${preference_suite}"
sandbox-exec -f "${profile_path}" \
  "${preference_probe_bin}" verify "${preference_suite}"

sandbox-exec -f "${profile_path}" "${probe_bin}" \
  "${golden_path}" \
  "${private_path}" \
  "${outside_write_path}" \
  "${private_write_path}" \
  "${unix_socket_path}" \
  "${sysv_key}"
