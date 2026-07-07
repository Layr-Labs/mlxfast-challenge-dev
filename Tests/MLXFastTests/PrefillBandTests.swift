import Foundation
@testable import MLXFastCore
import Testing

// Validates the prefill acceptance band: drop the single slowest sample, average
// the rest into B, accept iff every retained sample is within [B*(1-down), B*(1+up)].
// Numbers anchored on the real tenki fresh-VM run 28893815980.

@Test
func prefillBandAcceptsRealFreshVMSamples() {
    // 6 fresh-VM prefills; run 5 (0.01532) is the +44% outlier.
    let r = PrefillBand.evaluate(samples: [
        0.010556244302734375, 0.01071322314453125, 0.010636167642578125,
        0.010626223958984375, 0.01532235343359375, 0.01049330069921875,
    ])
    #expect(r.passed)
    #expect(abs(r.accepted - 0.0106048) < 1e-5)  // mean of the 5 after dropping the spike
}

@Test
func prefillBandFailsWhenASecondSlowSpikeSurvives() {
    // Two slow spikes: dropping one leaves the other above the +2% band.
    let r = PrefillBand.evaluate(samples: [0.0106, 0.0107, 0.0106, 0.020, 0.021])
    #expect(!r.passed)
    #expect(r.reason.contains("slowdown"))
}

@Test
func prefillBandFailsOnLuckyFastReading() {
    // A lucky-fast sample is NOT dropped (only the slowest is); it drags B down
    // and trips the band, which is the intended "got lucky -> fail" behavior.
    let r = PrefillBand.evaluate(samples: [0.0106, 0.0107, 0.0106, 0.0106, 0.006])
    #expect(!r.passed)
}

@Test
func prefillBandAcceptsTightSamplesAndOneSlowOutlier() {
    #expect(PrefillBand.evaluate(samples: [0.0106, 0.0107, 0.0105, 0.0106, 0.0106]).passed)
    // One slow outlier is dropped; the rest are tight -> accept.
    #expect(PrefillBand.evaluate(samples: [0.0106, 0.0107, 0.0106, 0.0105, 0.030]).passed)
}

@Test
func prefillBandRejectsTooFewSamples() {
    #expect(!PrefillBand.evaluate(samples: [0.0106, 0.0106]).passed)
    #expect(!PrefillBand.evaluate(samples: []).passed)
}

@Test
func prefillBandRejectsNonFiniteOrNonPositive() {
    #expect(!PrefillBand.evaluate(samples: [0.0106, 0.0, 0.0106]).passed)
    #expect(!PrefillBand.evaluate(samples: [0.0106, .nan, 0.0106]).passed)
}

@Test
func prefillBandDefaultTolerancesMatchConstants() {
    #expect(MLXFastConstants.prefillBandUpTolerance == 0.02)
    #expect(MLXFastConstants.prefillBandDownTolerance == 0.05)
    #expect(MLXFastConstants.prefillBandSampleCount == 5)
}
