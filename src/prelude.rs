//! Generic prelude.

pub use super::audionode::*;
pub use super::audiounit::*;
pub use super::combinator::*;
pub use super::delay::*;
pub use super::dynamics::*;
pub use super::envelope::*;
pub use super::feedback::*;
pub use super::filter::*;
pub use super::fir::*;
pub use super::math::*;
pub use super::moog::*;
pub use super::net::*;
pub use super::noise::*;
pub use super::oscillator::*;
pub use super::oversample::*;
pub use super::pan::*;
pub use super::sequencer::*;
pub use super::shape::*;
pub use super::signal::*;
pub use super::svf::*;
pub use super::wave::*;
pub use super::wavetable::*;
pub use super::*;

use num_complex::Complex64;
use std::sync::Arc;

// Combinator environment.
// We like to define all kinds of useful functions here.

// Import some typenum integers for reporting arities.
pub type U0 = numeric_array::typenum::U0;
pub type U1 = numeric_array::typenum::U1;
pub type U2 = numeric_array::typenum::U2;
pub type U3 = numeric_array::typenum::U3;
pub type U4 = numeric_array::typenum::U4;
pub type U5 = numeric_array::typenum::U5;
pub type U6 = numeric_array::typenum::U6;
pub type U7 = numeric_array::typenum::U7;
pub type U8 = numeric_array::typenum::U8;
pub type U9 = numeric_array::typenum::U9;
pub type U10 = numeric_array::typenum::U10;
pub type U11 = numeric_array::typenum::U11;
pub type U12 = numeric_array::typenum::U12;
pub type U13 = numeric_array::typenum::U13;
pub type U14 = numeric_array::typenum::U14;
pub type U15 = numeric_array::typenum::U15;
pub type U16 = numeric_array::typenum::U16;
pub type U17 = numeric_array::typenum::U17;
pub type U18 = numeric_array::typenum::U18;
pub type U19 = numeric_array::typenum::U19;
pub type U20 = numeric_array::typenum::U20;
pub type U21 = numeric_array::typenum::U21;
pub type U22 = numeric_array::typenum::U22;
pub type U23 = numeric_array::typenum::U23;
pub type U24 = numeric_array::typenum::U24;
pub type U25 = numeric_array::typenum::U25;
pub type U26 = numeric_array::typenum::U26;
pub type U27 = numeric_array::typenum::U27;
pub type U28 = numeric_array::typenum::U28;
pub type U29 = numeric_array::typenum::U29;
pub type U30 = numeric_array::typenum::U30;
pub type U31 = numeric_array::typenum::U31;
pub type U32 = numeric_array::typenum::U32;
pub type U33 = numeric_array::typenum::U33;
pub type U34 = numeric_array::typenum::U34;
pub type U35 = numeric_array::typenum::U35;
pub type U36 = numeric_array::typenum::U36;
pub type U37 = numeric_array::typenum::U37;
pub type U38 = numeric_array::typenum::U38;
pub type U39 = numeric_array::typenum::U39;
pub type U40 = numeric_array::typenum::U40;
pub type U41 = numeric_array::typenum::U41;
pub type U42 = numeric_array::typenum::U42;
pub type U43 = numeric_array::typenum::U43;
pub type U44 = numeric_array::typenum::U44;
pub type U45 = numeric_array::typenum::U45;
pub type U46 = numeric_array::typenum::U46;
pub type U47 = numeric_array::typenum::U47;
pub type U48 = numeric_array::typenum::U48;
pub type U49 = numeric_array::typenum::U49;
pub type U50 = numeric_array::typenum::U50;
pub type U51 = numeric_array::typenum::U51;
pub type U52 = numeric_array::typenum::U52;
pub type U53 = numeric_array::typenum::U53;
pub type U54 = numeric_array::typenum::U54;
pub type U55 = numeric_array::typenum::U55;
pub type U56 = numeric_array::typenum::U56;
pub type U57 = numeric_array::typenum::U57;
pub type U58 = numeric_array::typenum::U58;
pub type U59 = numeric_array::typenum::U59;
pub type U60 = numeric_array::typenum::U60;
pub type U61 = numeric_array::typenum::U61;
pub type U62 = numeric_array::typenum::U62;
pub type U63 = numeric_array::typenum::U63;
pub type U64 = numeric_array::typenum::U64;
pub type U65 = numeric_array::typenum::U65;
pub type U66 = numeric_array::typenum::U66;
pub type U67 = numeric_array::typenum::U67;
pub type U68 = numeric_array::typenum::U68;
pub type U69 = numeric_array::typenum::U69;
pub type U70 = numeric_array::typenum::U70;
pub type U71 = numeric_array::typenum::U71;
pub type U72 = numeric_array::typenum::U72;
pub type U73 = numeric_array::typenum::U73;
pub type U74 = numeric_array::typenum::U74;
pub type U75 = numeric_array::typenum::U75;
pub type U76 = numeric_array::typenum::U76;
pub type U77 = numeric_array::typenum::U77;
pub type U78 = numeric_array::typenum::U78;
pub type U79 = numeric_array::typenum::U79;
pub type U80 = numeric_array::typenum::U80;
pub type U81 = numeric_array::typenum::U81;
pub type U82 = numeric_array::typenum::U82;
pub type U83 = numeric_array::typenum::U83;
pub type U84 = numeric_array::typenum::U84;
pub type U85 = numeric_array::typenum::U85;
pub type U86 = numeric_array::typenum::U86;
pub type U87 = numeric_array::typenum::U87;
pub type U88 = numeric_array::typenum::U88;
pub type U89 = numeric_array::typenum::U89;
pub type U90 = numeric_array::typenum::U80;
pub type U91 = numeric_array::typenum::U91;
pub type U92 = numeric_array::typenum::U92;
pub type U93 = numeric_array::typenum::U93;
pub type U94 = numeric_array::typenum::U94;
pub type U95 = numeric_array::typenum::U95;
pub type U96 = numeric_array::typenum::U96;
pub type U97 = numeric_array::typenum::U97;
pub type U98 = numeric_array::typenum::U98;
pub type U99 = numeric_array::typenum::U99;
pub type U100 = numeric_array::typenum::U100;
pub type U101 = numeric_array::typenum::U101;
pub type U102 = numeric_array::typenum::U102;
pub type U103 = numeric_array::typenum::U103;
pub type U104 = numeric_array::typenum::U104;
pub type U105 = numeric_array::typenum::U105;
pub type U106 = numeric_array::typenum::U106;
pub type U107 = numeric_array::typenum::U107;
pub type U108 = numeric_array::typenum::U108;
pub type U109 = numeric_array::typenum::U109;
pub type U110 = numeric_array::typenum::U110;
pub type U111 = numeric_array::typenum::U111;
pub type U112 = numeric_array::typenum::U112;
pub type U113 = numeric_array::typenum::U113;
pub type U114 = numeric_array::typenum::U114;
pub type U115 = numeric_array::typenum::U115;
pub type U116 = numeric_array::typenum::U116;
pub type U117 = numeric_array::typenum::U117;
pub type U118 = numeric_array::typenum::U118;
pub type U119 = numeric_array::typenum::U119;
pub type U120 = numeric_array::typenum::U120;
pub type U121 = numeric_array::typenum::U121;
pub type U122 = numeric_array::typenum::U122;
pub type U123 = numeric_array::typenum::U123;
pub type U124 = numeric_array::typenum::U124;
pub type U125 = numeric_array::typenum::U125;
pub type U126 = numeric_array::typenum::U126;
pub type U127 = numeric_array::typenum::U127;
pub type U128 = numeric_array::typenum::U128;

/// Constant node. The constant can be scalar, tuple, or a Frame.
/// Synonymous with [`dc`].
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// constant(440.0) >> sine::<f32>();
/// ```
#[inline]
pub fn constant<T: Float, X: ConstantFrame<Sample = T>>(x: X) -> An<Constant<X::Size, T>>
where
    X::Size: Size<T>,
{
    An(Constant::new(x.convert()))
}

/// Constant node. The constant can be scalar, tuple, or a Frame.
/// Synonymous with [`constant`].
/// (DC stands for "direct current", which is an electrical engineering term used with signals.)
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// dc((220.0, 440.0)) >> (sine::<f64>() + sine());
/// ```
#[inline]
pub fn dc<T: Float, X: ConstantFrame<Sample = T>>(x: X) -> An<Constant<X::Size, T>>
where
    X::Size: Size<T>,
{
    An(Constant::new(x.convert()))
}

/// Tagged constant. Outputs the (scalar) value of the tag.
/// - Output 0: value
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// pass() & tag::<f32>(0, 0.2) * chorus(0, 0.015, 0.005, 0.5);
/// ```
#[inline]
pub fn tag<T: Float>(tag: Tag, value: T) -> An<Tagged<T>> {
    An(Tagged::new(tag, value))
}

/// Zero generator.
/// - Output 0: zero
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// zero::<f64>() >> pluck(220.0, db_amp(-6.0), 0.5);
/// ```
#[inline]
pub fn zero<T: Float>() -> An<Constant<U1, T>> {
    dc(T::new(0))
}

/// Multichannel zero generator.
/// - Output(s): zero
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// multizero::<U2, f64>() >> (pluck(220.0, db_amp(-6.0), 0.5) | pluck(220.0, db_amp(-6.0), 0.5));
/// ```
#[inline]
pub fn multizero<N: Size<T>, T: Float>() -> An<Constant<N, T>> {
    An(Constant::new(Frame::splat(T::zero())))
}

/// Mono pass-through.
/// - Input 0: signal
/// - Output 0: signal
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// pass::<f64>() & 0.2 * feedback(delay(1.0) * db_amp(-3.0));
/// ```
#[inline]
pub fn pass<T: Float>() -> An<Pass<T>> {
    An(Pass::new())
}

/// Multichannel pass-through.
/// - Input(s): signal
/// - Output(s): signal
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// multipass::<U2, f64>() & 0.2 * feedback((delay(1.0) | delay(1.0)) * db_amp(-3.0));
/// ```
#[inline]
pub fn multipass<N: Size<T>, T: Float>() -> An<MultiPass<N, T>> {
    An(MultiPass::new())
}

/// Timer node. A node with no inputs or outputs that presents time as a parameter.
/// It can be added to any node by stacking.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// timer::<f32>(0) | lfo(|t: f32| 1.0 / (1.0 + t));
/// ```
#[inline]
pub fn timer<T: Float>(tag: Tag) -> An<Timer<T>> {
    An(Timer::new(DEFAULT_SR, tag))
}

/// Monitor node. Passes through input and retains the latest input as a parameter.
/// - Input 0: signal
/// - Output 0: signal
#[inline]
pub fn monitor<T: Real>(meter: Meter, tag: Tag) -> An<Monitor<T>> {
    An(Monitor::new(tag, DEFAULT_SR, meter))
}

/// Meter node.
/// Outputs a summary of the input according to the chosen metering mode.
/// - Input 0: signal
/// - Output 0: summary
#[inline]
pub fn meter<T: Real>(meter: Meter) -> An<MeterNode<T>> {
    An(MeterNode::new(DEFAULT_SR, meter))
}

/// Mono sink. Input is discarded.
/// -Input 0: signal
#[inline]
pub fn sink<T: Float>() -> An<Sink<U1, T>> {
    An(Sink::new())
}

/// Multichannel sink. Inputs are discarded.
/// -Input(s): signal
#[inline]
pub fn multisink<N: Size<T>, T: Float>() -> An<Sink<N, T>> {
    An(Sink::new())
}

/// Swap stereo channels.
/// - Input 0: left channel
/// - Input 1: right channel
/// - Output 0: right channel
/// - Output 1: left channel
#[inline]
pub fn swap<T: Float>() -> An<Swap<T>> {
    An(Swap::new())
}

/// Sine oscillator.
/// - Input 0: frequency (Hz)
/// - Output 0: sine wave
#[inline]
pub fn sine<T: Real>() -> An<Sine<T>> {
    An(Sine::new(DEFAULT_SR))
}

/// Fixed sine oscillator at `f` Hz.
/// - Output 0: sine wave
#[inline]
pub fn sine_hz<T: Real>(f: T) -> An<Pipe<T, Constant<U1, T>, Sine<T>>> {
    constant(f) >> sine()
}

/// Add constant to signal.
/// - Input(s): signal
/// - Output(s): signal plus constant
#[inline]
pub fn add<X: ConstantFrame>(
    x: X,
) -> An<
    Binop<
        X::Sample,
        FrameAdd<X::Size, X::Sample>,
        MultiPass<X::Size, X::Sample>,
        Constant<X::Size, X::Sample>,
    >,
>
where
    X::Size: Size<X::Sample> + Add<U0>,
    <X::Size as Add<U0>>::Output: Size<X::Sample>,
{
    An(MultiPass::<X::Size, X::Sample>::new()) + dc(x)
}

/// Subtract constant from signal.
/// - Input(s): signal
/// - Output(s): signal minus constant
#[inline]
pub fn sub<X: ConstantFrame>(
    x: X,
) -> An<
    Binop<
        X::Sample,
        FrameSub<X::Size, X::Sample>,
        MultiPass<X::Size, X::Sample>,
        Constant<X::Size, X::Sample>,
    >,
>
where
    X::Size: Size<X::Sample> + Add<U0>,
    <X::Size as Add<U0>>::Output: Size<X::Sample>,
{
    An(MultiPass::<X::Size, X::Sample>::new()) - dc(x)
}

/// Multiply signal with constant.
/// - Input(s): signal
/// - Output(s): signal times constant
#[inline]
pub fn mul<X: ConstantFrame>(
    x: X,
) -> An<
    Binop<
        X::Sample,
        FrameMul<X::Size, X::Sample>,
        MultiPass<X::Size, X::Sample>,
        Constant<X::Size, X::Sample>,
    >,
>
where
    X::Size: Size<X::Sample> + Add<U0>,
    <X::Size as Add<U0>>::Output: Size<X::Sample>,
{
    An(MultiPass::<X::Size, X::Sample>::new()) * dc(x)
}

/// Butterworth lowpass filter (2nd order).
/// - Input 0: audio
/// - Input 1: cutoff frequency (Hz)
/// - Output 0: filtered audio
#[inline]
pub fn butterpass<T: Float, F: Real>() -> An<ButterLowpass<T, F, U2>> {
    An(ButterLowpass::new(convert(DEFAULT_SR), F::new(440)))
}

/// Butterworth lowpass filter (2nd order) with fixed cutoff frequency `f` Hz.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn butterpass_hz<T: Float, F: Real>(f: T) -> An<ButterLowpass<T, F, U1>> {
    An(ButterLowpass::new(convert(DEFAULT_SR), convert(f)))
}

/// One-pole lowpass filter (1st order).
/// - Input 0: audio
/// - Input 1: cutoff frequency (Hz)
/// - Output 0: filtered audio
#[inline]
pub fn lowpole<T: Float, F: Real>() -> An<Lowpole<T, F, U2>> {
    An(Lowpole::new(convert(DEFAULT_SR), F::new(440)))
}

/// One-pole lowpass filter (1st order) with fixed cutoff frequency `f` Hz.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn lowpole_hz<T: Float, F: Real>(f: T) -> An<Lowpole<T, F, U1>> {
    An(Lowpole::new(DEFAULT_SR, convert(f)))
}

/// Allpass filter with a configurable delay (delay > 0) in samples at DC.
/// - Input 0: audio
/// - Input 1: delay in samples
/// - Output 0: filtered audio
#[inline]
pub fn allpole<T: Float, F: Float>() -> An<Allpole<T, F, U2>> {
    An(Allpole::new(DEFAULT_SR, F::new(1)))
}

/// Allpass filter with delay (delay > 0) in samples at DC.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn allpole_delay<T: Float, F: Float>(delay_in_samples: F) -> An<Allpole<T, F, U1>> {
    An(Allpole::new(DEFAULT_SR, delay_in_samples))
}

/// One-pole, one-zero highpass filter (1st order).
/// - Input 0: audio
/// - Input 1: cutoff frequency (Hz)
/// - Output 0: filtered audio
#[inline]
pub fn highpole<T: Float, F: Real>() -> An<Highpole<T, F, U2>> {
    An(Highpole::new(DEFAULT_SR, F::new(440)))
}

/// One-pole, one-zero highpass filter (1st order) with fixed cutoff frequency f.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn highpole_hz<T: Float, F: Real>(f: T) -> An<Highpole<T, F, U1>> {
    An(Highpole::new(DEFAULT_SR, convert(f)))
}

/// Constant-gain bandpass resonator.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Input 2: bandwidth (Hz)
/// - Output 0: filtered audio
#[inline]
pub fn resonator<T: Float, F: Real>() -> An<Resonator<T, F, U3>> {
    An(Resonator::new(
        convert(DEFAULT_SR),
        F::new(440),
        F::new(110),
    ))
}

/// Constant-gain bandpass resonator with fixed `center` frequency (Hz) and `bandwidth` (Hz).
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn resonator_hz<T: Float, F: Real>(center: T, bandwidth: T) -> An<Resonator<T, F, U1>> {
    An(Resonator::new(
        convert(DEFAULT_SR),
        convert(center),
        convert(bandwidth),
    ))
}

/// Moog resonant lowpass filter.
/// - Input 0: input signal
/// - Input 1: cutoff frequency (Hz)
/// - Input 2: Q
/// - Output 0: filtered signal
#[inline]
pub fn moog<T: Float, F: Real>() -> An<Moog<T, F, U3>> {
    An(Moog::new(
        convert(DEFAULT_SR),
        F::new(1000),
        F::from_f64(0.1),
    ))
}

/// Moog resonant lowpass filter with fixed Q.
/// - Input 0: input signal
/// - Input 1: cutoff frequency (Hz)
/// - Output 0: filtered signal
#[inline]
pub fn moog_q<T: Float, F: Real>(
    q: T,
) -> An<Pipe<T, Stack<T, MultiPass<U2, T>, Constant<U1, T>>, Moog<T, F, U3>>> {
    (multipass::<U2, T>() | dc(q)) >> An(Moog::new(convert(DEFAULT_SR), F::new(1000), convert(q)))
}

/// Moog resonant lowpass filter with fixed cutoff frequency and Q.
/// - Input 0: input signal
/// - Output 0: filtered signal
#[inline]
pub fn moog_hz<T: Float, F: Real>(frequency: F, q: F) -> An<Moog<T, F, U1>> {
    An(Moog::new(convert(DEFAULT_SR), frequency, q))
}

/// Control envelope from time-varying function `f(t)` with `t` in seconds.
/// Spaces samples using pseudorandom jittering.
/// Synonymous with [`fn@lfo`].
/// - Output(s): envelope linearly interpolated from samples at 2 ms intervals (average).
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// envelope(|t: f32| (sin_hz(1.0, t), cos_hz(1.0, t))) * (brown::<f32, f32>() | white());
/// ```
#[inline]
pub fn envelope<T, F, E, R>(f: E) -> An<Envelope<T, F, E, R>>
where
    T: Float,
    F: Float,
    E: Fn(F) -> R,
    R: ConstantFrame<Sample = F>,
    R::Size: Size<F>,
    R::Size: Size<T>,
{
    // Signals containing frequencies no greater than about 20 Hz would be considered control rate.
    // Therefore, sampling at 500 Hz means these signals are fairly well represented.
    // While we represent time in double precision internally, it is often okay to use single precision
    // in envelopes, as local component time typically does not get far from origin.
    An(Envelope::new(F::from_f64(0.002), DEFAULT_SR, f))
}

/// Control envelope from time-varying function `f(t)` with `t` in seconds.
/// Spaces samples using pseudorandom jittering.
/// Synonymous with [`fn@envelope`].
/// - Output(s): envelope linearly interpolated from samples at 2 ms intervals (average).
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// lfo(|t: f32| exp(-t)) * white::<f32>();
/// ```
#[inline]
pub fn lfo<T, F, E, R>(f: E) -> An<Envelope<T, F, E, R>>
where
    T: Float,
    F: Float,
    E: Fn(F) -> R,
    R: ConstantFrame<Sample = F>,
    R::Size: Size<F>,
    R::Size: Size<T>,
{
    An(Envelope::new(F::from_f64(0.002), DEFAULT_SR, f))
}

/// Control envelope from time-varying, input dependent function `f(t, x)` with `t` in seconds.
/// Spaces samples using pseudorandom jittering.
/// Synonymous with `lfo2`.
/// - Input 0: x
/// - Output(s): envelope linearly interpolated from samples at 2 ms intervals (average).
#[inline]
pub fn envelope2<T, F, E, R>(f: E) -> An<Envelope2<T, F, E, R>>
where
    T: Float,
    F: Float,
    E: Fn(F, F) -> R,
    R: ConstantFrame<Sample = F>,
    R::Size: Size<F>,
    R::Size: Size<T>,
{
    // Signals containing frequencies no greater than about 20 Hz would be considered control rate.
    // Therefore, sampling at 500 Hz means these signals are fairly well represented.
    // While we represent time in double precision internally, it is often okay to use single precision
    // in envelopes, as local component time typically does not get far from origin.
    An(Envelope2::new(F::from_f64(0.002), DEFAULT_SR, f))
}

/// Control envelope from time-varying, input dependent function `f(t, x)` with `t` in seconds.
/// Spaces samples using pseudorandom jittering.
/// Synonymous with `envelope2`.
/// - Input 0: x
/// - Output(s): envelope linearly interpolated from samples at 2 ms intervals (average).
#[inline]
pub fn lfo2<T, F, E, R>(f: E) -> An<Envelope2<T, F, E, R>>
where
    T: Float,
    F: Float,
    E: Fn(F, F) -> R,
    R: ConstantFrame<Sample = F>,
    R::Size: Size<F>,
    R::Size: Size<T>,
{
    An(Envelope2::new(F::from_f64(0.002), DEFAULT_SR, f))
}

/// Control envelope from time-varying, input dependent function `f(t, x, y)` with `t` in seconds.
/// Spaces samples using pseudorandom jittering.
/// Synonymous with `lfo3`.
/// - Input 0: x
/// - Input 1: y
/// - Output(s): envelope linearly interpolated from samples at 2 ms intervals (average).
#[inline]
pub fn envelope3<T, F, E, R>(f: E) -> An<Envelope3<T, F, E, R>>
where
    T: Float,
    F: Float,
    E: Fn(F, F, F) -> R,
    R: ConstantFrame<Sample = F>,
    R::Size: Size<F>,
    R::Size: Size<T>,
{
    // Signals containing frequencies no greater than about 20 Hz would be considered control rate.
    // Therefore, sampling at 500 Hz means these signals are fairly well represented.
    // While we represent time in double precision internally, it is often okay to use single precision
    // in envelopes, as local component time typically does not get far from origin.
    An(Envelope3::new(F::from_f64(0.002), DEFAULT_SR, f))
}

/// Control envelope from time-varying, input dependent function `f(t, x, y)` with `t` in seconds.
/// Spaces samples using pseudorandom jittering.
/// Synonymous with `envelope3`.
/// - Output(s): envelope linearly interpolated from samples at 2 ms intervals (average).
#[inline]
pub fn lfo3<T, F, E, R>(f: E) -> An<Envelope3<T, F, E, R>>
where
    T: Float,
    F: Float,
    E: Fn(F, F, F) -> R,
    R: ConstantFrame<Sample = F>,
    R::Size: Size<F>,
    R::Size: Size<T>,
{
    An(Envelope3::new(F::from_f64(0.002), DEFAULT_SR, f))
}

/// Maximum Length Sequence noise generator from an `n`-bit sequence (1 <= `n` <= 31).
/// - Output 0: repeating white noise sequence of only -1 and 1 values.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// mls_bits::<f32>(31);
/// ```
#[inline]
pub fn mls_bits<T: Float>(n: i64) -> An<Mls<T>> {
    An(Mls::new(MlsState::new(n as u32)))
}

/// Default Maximum Length Sequence noise generator.
/// - Output 0: repeating white noise sequence of only -1 and 1 values.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// mls::<f32>();
/// ```
#[inline]
pub fn mls<T: Float>() -> An<Mls<T>> {
    mls_bits(29)
}

/// White noise generator.
/// Synonymous with [`fn@white`].
/// - Output 0: white noise.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// noise::<f32>();
/// ```
#[inline]
pub fn noise<T: Float>() -> An<Noise<T>> {
    An(Noise::new())
}

/// White noise generator.
/// Synonymous with [`fn@noise`].
/// - Output 0: white noise.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// white::<f32>();
/// ```
#[inline]
pub fn white<T: Float>() -> An<Noise<T>> {
    An(Noise::new())
}

/// FIR filter.
/// - Input 0: signal.
/// - Output 0: filtered signal.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// fir(Frame::<f64, _>::from([0.5, 1.0, 0.5]));
/// ```
#[inline]
pub fn fir<X: ConstantFrame>(weights: X) -> An<Fir<X::Sample, X::Size>> {
    An(Fir::new(weights))
}

/// Single sample delay.
/// - Input 0: signal.
/// - Output 0: delayed signal.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// tick::<f64>() & pass();
/// ```
#[inline]
pub fn tick<T: Float>() -> An<Tick<U1, T>> {
    An(Tick::new(convert(DEFAULT_SR)))
}

/// Multichannel single sample delay.
/// - Inputs: signal.
/// - Outputs: delayed signal.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// multitick::<U2, f32>();
/// ```
#[inline]
pub fn multitick<N: Size<T>, T: Float>() -> An<Tick<N, T>> {
    An(Tick::new(convert(DEFAULT_SR)))
}

/// Fixed delay of `t` seconds.
/// Delay time is rounded to the nearest sample.
/// - Input 0: signal.
/// - Output 0: delayed signal.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// delay::<f32>(1.0);
/// ```
#[inline]
pub fn delay<T: Float>(t: f64) -> An<Delay<T>> {
    An(Delay::new(t, DEFAULT_SR))
}

/// Tapped delay line with cubic interpolation.
/// Minimum and maximum delay times are in seconds.
/// - Input 0: signal.
/// - Input 1: delay time in seconds.
/// - Output 0: delayed signal.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// pass::<f32>() & (pass() | lfo(|t| lerp11(0.0, 0.1, spline_noise(0, t)))) >> tap(0.0, 0.1);
/// ```
#[inline]
pub fn tap<T: Float>(min_delay: f64, max_delay: f64) -> An<Tap<U1, T>> {
    An(Tap::new(DEFAULT_SR, min_delay, max_delay))
}

/// Tapped delay line with cubic interpolation.
/// The number of taps is `N`.
/// Minimum and maximum delay times are in seconds.
/// - Input 0: signal.
/// - Inputs 1...N: delay time in seconds.
/// - Output 0: delayed signal.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// (pass() | lfo(|t| (lerp11(0.0, 0.1, spline_noise(0, t)), lerp11(0.1, 0.2, spline_noise(1, t))))) >> multitap::<U2, f64>(0.0, 0.2);
/// ```
#[inline]
pub fn multitap<N, T>(min_delay: f64, max_delay: f64) -> An<Tap<N, T>>
where
    T: Float,
    N: Size<T> + Add<U1>,
    <N as Add<U1>>::Output: Size<T>,
{
    An(Tap::new(DEFAULT_SR, min_delay, max_delay))
}

/// 2x oversample enclosed `node`.
/// - Inputs and outputs: from `node`.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// let f: f64 = 440.0;
/// let m: f64 = 1.0;
/// oversample(sine_hz(f) * f * m + f >> sine());
/// ```
#[inline]
pub fn oversample<T, X>(node: An<X>) -> An<Oversampler<T, X>>
where
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
    X::Inputs: Size<Frame<T, U128>>,
    X::Outputs: Size<Frame<T, U128>>,
{
    An(Oversampler::new(DEFAULT_SR, node.0))
}

/// Mix output of enclosed circuit `node` back to its input.
/// Feedback circuit `node` must have an equal number of inputs and outputs.
/// - Inputs: input signal.
/// - Outputs: `node` output signal.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// pass() & feedback(delay(1.0) >> lowpass_hz::<f64, f64>(1000.0, 1.0));
/// ```
#[inline]
pub fn feedback<N, T, X>(node: An<X>) -> An<Feedback<N, T, X, FrameId<N, T>>>
where
    N: Size<T>,
    T: Float,
    X: AudioNode<Sample = T, Inputs = N, Outputs = N>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
{
    An(Feedback::new(node.0, FrameId::new()))
}

/// Transform channels freely. Accounted as non-linear processing for signal flow.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// map(|i: &Frame<f64, U2>| max(i[0], i[1]));
/// ```
#[inline]
pub fn map<T, M, I, O>(f: M) -> An<Map<T, M, I, O>>
where
    T: Float,
    M: Fn(&Frame<T, I>) -> O,
    I: Size<T>,
    O: ConstantFrame<Sample = T>,
    O::Size: Size<T>,
{
    An(Map::new(f, Routing::Arbitrary))
}

/// Keeps a signal zero centered.
/// Filter cutoff `c` Hz is usually somewhere below the audible range.
/// The default blocker cutoff is 10 Hz.
/// - Input 0: signal
/// - Output 0: filtered signal
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// dcblock_hz::<f64, f64>(8.0);
/// ```
#[inline]
pub fn dcblock_hz<T: Float, F: Real>(c: F) -> An<DCBlock<T, F>> {
    An(DCBlock::new(DEFAULT_SR, c))
}

/// Keeps a signal zero centered. The cutoff of the filter is 10 Hz.
/// - Input 0: signal
/// - Output 0: filtered signal
#[inline]
pub fn dcblock<T: Float, F: Real>() -> An<DCBlock<T, F>> {
    An(DCBlock::new(DEFAULT_SR, F::new(10)))
}

/// Apply 10 ms of fade-in to signal at time zero.
/// - Input 0: input signal
/// - Output 0: signal with fade-in
#[inline]
pub fn declick<T: Float, F: Real>() -> An<Declick<T, F>> {
    An(Declick::new(DEFAULT_SR, F::from_f64(0.010)))
}

/// Apply `t` seconds of fade-in to signal at time zero.
/// - Input 0: input signal
/// - Output 0: signal with fade-in
#[inline]
pub fn declick_s<T: Float, F: Real>(t: F) -> An<Declick<T, F>> {
    An(Declick::new(DEFAULT_SR, t))
}

/// Shape signal with a waveshaper function.
/// - Input 0: input signal
/// - Output 0: shaped signal
#[inline]
pub fn shape_fn<T: Float, S: Fn(T) -> T>(f: S) -> An<ShaperFn<T, S>> {
    An(ShaperFn::new(f))
}

/// Shape signal.
/// - Input 0: input signal
/// - Output 0: shaped signal
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// shape::<f64>(Shape::Tanh(1.0));
/// ```
#[inline]
pub fn shape<T: Real>(mode: Shape<T>) -> An<Shaper<T>> {
    An(Shaper::new(mode))
}

/// Clip signal to -1...1.
/// - Input 0: input signal
/// - Output 0: clipped signal
#[inline]
pub fn clip<T: Real>() -> An<Shaper<T>> {
    An(Shaper::<T>::new(Shape::Clip))
}

/// Clip signal to `minimum`...`maximum`.
/// - Input 0: input signal
/// - Output 0: clipped signal
#[inline]
pub fn clip_to<T: Real>(minimum: T, maximum: T) -> An<Shaper<T>> {
    An(Shaper::<T>::new(Shape::ClipTo(minimum, maximum)))
}

/// Equal power mono-to-stereo panner.
/// - Input 0: input signal
/// - Input 1: pan in -1...1 (left to right).
/// - Output 0: left channel
/// - Output 1: right channel
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// (noise() | sine_hz(0.5)) >> panner::<f64>();
/// ```
#[inline]
pub fn panner<T: Real>() -> An<Panner<T, U2>> {
    An(Panner::new(T::zero()))
}

/// Fixed equal power mono-to-stereo panner with pan value in -1...1.
/// - Input 0: input signal
/// - Output 0: left channel
/// - Output 1: right channel
#[inline]
pub fn pan<T: Real>(pan: T) -> An<Panner<T, U1>> {
    An(Panner::new(pan))
}

/// Parameter follower filter with halfway response time `t` seconds.
/// - Input 0: input signal
/// - Output 0: smoothed signal
#[inline]
pub fn follow<T: Float, F: Real, S: ScalarOrPair<Sample = F>>(t: S) -> An<AFollow<T, F, S>> {
    An(AFollow::new(DEFAULT_SR, t))
}

/// Look-ahead limiter with `(attack, release)` times in seconds.
/// Look-ahead is equal to the attack time.
#[inline]
pub fn limiter<T: Real, S: ScalarOrPair<Sample = T>>(time: S) -> An<Limiter<T, U1, S>> {
    An(Limiter::new(DEFAULT_SR, time))
}

/// Stereo look-ahead limiter with `(attack, release)` times in seconds.
/// Look-ahead is equal to the attack time.
#[inline]
pub fn limiter_stereo<T: Real, S: ScalarOrPair<Sample = T>>(time: S) -> An<Limiter<T, U2, S>> {
    An(Limiter::new(DEFAULT_SR, time))
}

/// Pinking filter.
#[inline]
pub fn pinkpass<T: Float, F: Float>() -> An<Pinkpass<T, F>> {
    An(Pinkpass::new(DEFAULT_SR))
}

/// Pink noise.
#[inline]
pub fn pink<T: Float, F: Float>() -> An<Pipe<T, Noise<T>, Pinkpass<T, F>>> {
    white() >> pinkpass::<T, F>()
}

/// Brown noise.
#[inline]
pub fn brown<T: Float, F: Real>(
) -> An<Pipe<T, Noise<T>, Binop<T, FrameMul<U1, T>, Lowpole<T, F, U1>, Constant<U1, T>>>> {
    // Empirical normalization factor.
    white() >> lowpole_hz::<T, F>(T::from_f64(10.0)) * dc(T::from_f64(13.7))
}

/// Frequency detector.
#[inline]
pub fn goertzel<T: Float, F: Real>() -> An<Goertzel<T, F>> {
    An(Goertzel::new(DEFAULT_SR))
}

/// Frequency detector of frequency `f` Hz.
#[inline]
pub fn goertzel_hz<T: Float, F: Real>(
    f: T,
) -> An<Pipe<T, Stack<T, Pass<T>, Constant<U1, T>>, Goertzel<T, F>>> {
    (pass() | constant(f)) >> goertzel::<T, F>()
}

/// Feedback delay network.
/// Mix output of enclosed circuit `x` back to its input.
/// The output is diffused with a Hadamard matrix for feedback.
/// Feedback circuit `x` must have an equal number of inputs and outputs.
/// - Inputs: input signal.
/// - Outputs: `x` output signal.
#[inline]
pub fn fdn<N, T, X>(x: An<X>) -> An<Feedback<N, T, X, FrameHadamard<N, T>>>
where
    N: Size<T>,
    T: Float,
    X: AudioNode<Sample = T, Inputs = N, Outputs = N>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
{
    An(Feedback::new(x.0, FrameHadamard::new()))
}

/// Bus `N` similar nodes from indexed generator `f`.
#[inline]
pub fn bus<N, T, X, F>(f: F) -> An<MultiBus<N, T, X>>
where
    T: Float,
    N: Size<T>,
    N: Size<X>,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
    F: Fn(i64) -> An<X>,
{
    assert!(N::USIZE > 0);
    let nodes = Frame::generate(|i| f(i as i64).0);
    An(MultiBus::new(nodes))
}

/// Bus `N` similar nodes from fractional generator `f`.
/// The fractional generator is given values in the range 0...1.
#[inline]
pub fn busf<N, T, X, F>(f: F) -> An<MultiBus<N, T, X>>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
    F: Fn(f64) -> An<X>,
{
    assert!(N::USIZE > 0);
    let nodes = Frame::generate(|i| {
        f(if N::USIZE > 1 {
            i as f64 / (N::USIZE - 1) as f64
        } else {
            0.5
        })
        .0
    });
    An(MultiBus::new(nodes))
}

/// Stack `N` similar nodes from indexed generator `f`.
#[inline]
pub fn stack<N, T, X, F>(f: F) -> An<MultiStack<N, T, X>>
where
    T: Float,
    N: Size<T>,
    N: Size<X>,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Mul<N>,
    X::Outputs: Size<T> + Mul<N>,
    <X::Inputs as Mul<N>>::Output: Size<T>,
    <X::Outputs as Mul<N>>::Output: Size<T>,
    F: Fn(i64) -> An<X>,
{
    assert!(N::USIZE > 0);
    let nodes = Frame::generate(|i| f(i as i64).0);
    An(MultiStack::new(nodes))
}

/// Stack `N` similar nodes from fractional generator `f`.
/// The fractional generator is given values in the range 0...1.
#[inline]
pub fn stackf<N, T, X, F>(f: F) -> An<MultiStack<N, T, X>>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Mul<N>,
    X::Outputs: Size<T> + Mul<N>,
    <X::Inputs as Mul<N>>::Output: Size<T>,
    <X::Outputs as Mul<N>>::Output: Size<T>,
    F: Fn(f64) -> An<X>,
{
    assert!(N::USIZE > 0);
    let nodes = Frame::generate(|i| {
        f(if N::USIZE > 1 {
            i as f64 / (N::USIZE - 1) as f64
        } else {
            0.5
        })
        .0
    });
    An(MultiStack::new(nodes))
}

/// Branch into `N` similar nodes from indexed generator `f`.
#[inline]
pub fn branch<N, T, X, F>(f: F) -> An<MultiBranch<N, T, X>>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T> + Mul<N>,
    <X::Outputs as Mul<N>>::Output: Size<T>,
    F: Fn(i64) -> An<X>,
{
    let nodes = Frame::generate(|i| f(i as i64).0);
    An(MultiBranch::new(nodes))
}

/// Branch into `N` similar nodes from fractional generator `f`.
/// The fractional generator is given values in the range 0...1.
#[inline]
pub fn branchf<N, T, X, F>(f: F) -> An<MultiBranch<N, T, X>>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T> + Mul<N>,
    <X::Outputs as Mul<N>>::Output: Size<T>,
    F: Fn(f64) -> An<X>,
{
    let nodes = Frame::generate(|i| {
        f(if N::USIZE > 1 {
            i as f64 / (N::USIZE - 1) as f64
        } else {
            0.5
        })
        .0
    });
    An(MultiBranch::new(nodes))
}

/// Mix together `N` similar nodes from indexed generator `f`.
#[inline]
pub fn sum<N, T, X, F>(f: F) -> An<Reduce<N, T, X, FrameAdd<X::Outputs, T>>>
where
    T: Float,
    N: Size<T>,
    N: Size<X>,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Mul<N>,
    X::Outputs: Size<T>,
    <X::Inputs as Mul<N>>::Output: Size<T>,
    F: Fn(i64) -> An<X>,
{
    let nodes = Frame::generate(|i| f(i as i64).0);
    An(Reduce::new(nodes, FrameAdd::new()))
}

/// Mix together `N` similar nodes from fractional generator `f`.
/// The fractional generator is given values in the range 0...1.
#[inline]
pub fn sumf<N, T, X, F>(f: F) -> An<Reduce<N, T, X, FrameAdd<X::Outputs, T>>>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Mul<N>,
    X::Outputs: Size<T>,
    <X::Inputs as Mul<N>>::Output: Size<T>,
    F: Fn(f64) -> An<X>,
{
    let nodes = Frame::generate(|i| {
        f(if N::USIZE > 1 {
            i as f64 / (N::USIZE - 1) as f64
        } else {
            0.5
        })
        .0
    });
    An(Reduce::new(nodes, FrameAdd::new()))
}

/// Chain together `N` similar nodes from indexed generator `f`.
#[inline]
pub fn pipe<N, T, X, F>(f: F) -> An<Chain<N, T, X>>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
    F: Fn(i64) -> An<X>,
{
    let nodes = Frame::generate(|i| f(i as i64).0);
    An(Chain::new(nodes))
}

/// Chain together `N` similar nodes from fractional generator `f`.
/// The fractional generator is given values in the range 0...1.
#[inline]
pub fn pipef<N, T, X, F>(f: F) -> An<Chain<N, T, X>>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
    F: Fn(f64) -> An<X>,
{
    let nodes = Frame::generate(|i| {
        f(if N::USIZE > 1 {
            i as f64 / (N::USIZE - 1) as f64
        } else {
            0.5
        })
        .0
    });
    An(Chain::new(nodes))
}

/// Split signal into N channels.
#[inline]
pub fn split<N, T>() -> An<Split<N, T>>
where
    N: Size<T>,
    T: Float,
{
    An(Split::new())
}

/// Split M channels into N branches. The output has M * N channels.
#[inline]
pub fn multisplit<M, N, T>() -> An<MultiSplit<M, N, T>>
where
    M: Size<T> + Mul<N>,
    N: Size<T>,
    <M as Mul<N>>::Output: Size<T>,
    T: Float,
{
    An(MultiSplit::new())
}

/// Average N channels into one. Inverse of `split`.
#[inline]
pub fn join<N, T>() -> An<Join<N, T>>
where
    T: Float,
    N: Size<T>,
{
    An(Join::new())
}

/// Average `N` branches of `M` channels into one branch with `M` channels.
/// The input has `M` * `N` channels. Inverse of `multisplit::<M, N>`.
#[inline]
pub fn multijoin<M, N, T>() -> An<MultiJoin<M, N, T>>
where
    N: Size<T>,
    M: Size<T> + Mul<N>,
    <M as Mul<N>>::Output: Size<T>,
    T: Float,
{
    An(MultiJoin::<M, N, T>::new())
}

/// Stereo reverb.
/// `wet` in 0...1 is balance of reverb mixed in, for example, 0.1.
/// `time` is approximate reverberation time to -60 dB in seconds.
pub fn reverb_stereo<T, F>(
    wet: T,
    time: f64,
) -> An<
    Bus<
        T,
        Pipe<
            T,
            Pipe<
                T,
                Pipe<
                    T,
                    MultiSplit<U2, U16, T>,
                    Feedback<
                        U32,
                        T,
                        MultiStack<
                            U32,
                            T,
                            Pipe<
                                T,
                                Pipe<T, Pipe<T, Delay<T>, Fir<T, U2>>, DCBlock<T, F>>,
                                Binop<T, FrameMul<U1, T>, MultiPass<U1, T>, Constant<U1, T>>,
                            >,
                        >,
                        FrameHadamard<U32, T>,
                    >,
                >,
                MultiJoin<U2, U16, T>,
            >,
            Binop<T, FrameMul<U2, T>, MultiPass<U2, T>, Constant<U2, T>>,
        >,
        Binop<T, FrameMul<U2, T>, MultiPass<U2, T>, Constant<U2, T>>,
    >,
>
where
    T: Float,
    F: Real,
{
    // TODO: This is the simplest possible structure, there's probably a lot of scope for improvement.

    // Optimized delay times for a 32-channel FDN from a legacy project.
    const DELAYS: [f64; 32] = [
        0.073904, 0.052918, 0.066238, 0.066387, 0.037783, 0.080073, 0.050961, 0.075900, 0.043646,
        0.072095, 0.056194, 0.045961, 0.058934, 0.068016, 0.047529, 0.058156, 0.072972, 0.036084,
        0.062715, 0.076377, 0.044339, 0.076725, 0.077884, 0.046126, 0.067741, 0.049800, 0.051709,
        0.082923, 0.070121, 0.079315, 0.055039, 0.081859,
    ];

    let a = T::from_f64(pow(db_amp(-60.0), 0.03 / time));

    let line = stack::<U32, T, _, _>(|i| {
        delay::<T>(DELAYS[i as usize])
            >> fir((T::from_f32(0.5), T::from_f32(0.5)))
            >> dcblock_hz::<T, F>(F::new(5))
            >> mul(a)
    });

    // The feedback structure.
    let reverb = fdn::<U32, T, _>(line);

    // Multiplex stereo into 32 channels, reverberate, then average them back.
    // Bus the reverb with the dry signal. Operator precedences work perfectly for us here.
    multisplit::<U2, U16, T>() >> reverb >> multijoin::<U2, U16, T>() >> mul((wet, wet))
        & mul((T::one() - wet, T::one() - wet))
}

/// Saw-like discrete summation formula oscillator.
/// - Input 0: frequency in Hz
/// - Input 1: roughness in 0...1 is the attenuation of successive partials.
/// - Output 0: DSF wave
pub fn dsf_saw<T: Real>() -> An<Dsf<T, U2>> {
    An(Dsf::new(DEFAULT_SR, T::new(1), T::from_f32(0.5)))
}

/// Saw-like discrete summation formula oscillator.
/// Roughness in 0...1 is the attenuation of successive partials.
/// - Input 0: frequency in Hz
/// - Output 0: DSF wave
pub fn dsf_saw_r<T: Real>(roughness: T) -> An<Dsf<T, U1>> {
    An(Dsf::new(DEFAULT_SR, T::new(1), roughness))
}

/// Square-like discrete summation formula oscillator.
/// - Input 0: frequency in Hz
/// - Input 1: roughness in 0...1 is the attenuation of successive partials.
/// - Output 0: DSF wave
pub fn dsf_square<T: Real>() -> An<Dsf<T, U2>> {
    An(Dsf::new(DEFAULT_SR, T::new(2), T::from_f32(0.5)))
}

/// Square-like discrete summation formula oscillator.
/// Roughness in 0...1 is the attenuation of successive partials.
/// - Input 0: frequency in Hz
/// - Output 0: DSF wave
pub fn dsf_square_r<T: Real>(roughness: T) -> An<Dsf<T, U1>> {
    An(Dsf::new(DEFAULT_SR, T::new(2), roughness))
}

/// Karplus-Strong plucked string oscillator with `frequency` in Hz.
/// High frequency damping is in 0...1.
/// - Input 0: string excitation
/// - Output 0: oscillator output
pub fn pluck<T: Float>(
    frequency: T,
    gain_per_second: T,
    high_frequency_damping: T,
) -> An<Pluck<T>> {
    An(Pluck::new(
        DEFAULT_SR,
        frequency,
        gain_per_second,
        high_frequency_damping,
    ))
}

/// Saw wave oscillator.
/// - Input 0: frequency in Hz
/// - Output 0: saw wave
#[inline]
pub fn saw<T: Float>() -> An<WaveSynth<'static, T, U1>> {
    An(WaveSynth::new(DEFAULT_SR, &SAW_TABLE))
}

/// Square wave oscillator.
/// - Input 0: frequency in Hz
/// - Output 0: square wave
#[inline]
pub fn square<T: Float>() -> An<WaveSynth<'static, T, U1>> {
    An(WaveSynth::new(DEFAULT_SR, &SQUARE_TABLE))
}

/// Triangle wave oscillator.
/// - Input 0: frequency in Hz
/// - Output 0: triangle wave
#[inline]
pub fn triangle<T: Float>() -> An<WaveSynth<'static, T, U1>> {
    An(WaveSynth::new(DEFAULT_SR, &TRIANGLE_TABLE))
}

/// Fixed saw wave oscillator at `f` Hz.
/// - Output 0: saw wave
#[inline]
pub fn saw_hz<T: Float>(f: T) -> An<Pipe<T, Constant<U1, T>, WaveSynth<'static, T, U1>>> {
    constant(f) >> saw()
}

/// Fixed square wave oscillator at `f` Hz.
/// - Output 0: square wave
#[inline]
pub fn square_hz<T: Float>(f: T) -> An<Pipe<T, Constant<U1, T>, WaveSynth<'static, T, U1>>> {
    constant(f) >> square()
}

/// Fixed triangle wave oscillator at `f` Hz.
/// - Output 0: triangle wave
#[inline]
pub fn triangle_hz<T: Float>(f: T) -> An<Pipe<T, Constant<U1, T>, WaveSynth<'static, T, U1>>> {
    constant(f) >> triangle()
}

/// Lowpass filter.
/// - Input 0: audio
/// - Input 1: cutoff frequency (Hz)
/// - Input 2: Q
/// - Output 0: filtered audio
#[inline]
pub fn lowpass<T: Float, F: Real>() -> An<Svf<T, F, LowpassMode<F>>> {
    An(Svf::new(
        LowpassMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: F::new(440),
            q: F::one(),
            gain: F::one(),
        },
    ))
}

/// Lowpass filter with cutoff frequency `f` Hz with Q value `q`.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn lowpass_hz<T: Float, F: Real>(f: T, q: T) -> An<FixedSvf<T, F, LowpassMode<F>>> {
    An(FixedSvf::new(
        LowpassMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: convert(f),
            q: convert(q),
            gain: F::one(),
        },
    ))
}

/// Lowpass filter with Q value `q`.
/// - Input 0: audio
/// - Input 1: cutoff frequency (Hz)
/// - Output 0: filtered audio
#[inline]
pub fn lowpass_q<T: Float, F: Real>(
    q: T,
) -> An<Pipe<T, Stack<T, MultiPass<U2, T>, Constant<U1, T>>, Svf<T, F, LowpassMode<F>>>> {
    (multipass::<U2, T>() | dc(q))
        >> An(Svf::new(
            LowpassMode::default(),
            &SvfParams {
                sample_rate: convert(DEFAULT_SR),
                cutoff: F::new(440),
                q: convert(q),
                gain: F::one(),
            },
        ))
}

/// Highpass filter.
/// - Input 0: audio
/// - Input 1: cutoff frequency (Hz)
/// - Input 2: Q
/// - Output 0: filtered audio
#[inline]
pub fn highpass<T: Float, F: Real>() -> An<Svf<T, F, HighpassMode<F>>> {
    An(Svf::new(
        HighpassMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: F::new(440),
            q: F::one(),
            gain: F::one(),
        },
    ))
}

/// Highpass filter with cutoff frequency `f` Hz with Q value `q`.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn highpass_hz<T: Float, F: Real>(f: T, q: T) -> An<FixedSvf<T, F, HighpassMode<F>>> {
    An(FixedSvf::new(
        HighpassMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: convert(f),
            q: convert(q),
            gain: F::one(),
        },
    ))
}

/// Highpass filter with Q value `q`.
/// - Input 0: audio
/// - Input 1: cutoff frequency (Hz)
/// - Output 0: filtered audio
#[inline]
pub fn highpass_q<T: Float, F: Real>(
    q: T,
) -> An<Pipe<T, Stack<T, MultiPass<U2, T>, Constant<U1, T>>, Svf<T, F, HighpassMode<F>>>> {
    (multipass::<U2, T>() | dc(q))
        >> An(Svf::new(
            HighpassMode::default(),
            &SvfParams {
                sample_rate: convert(DEFAULT_SR),
                cutoff: F::new(440),
                q: convert(q),
                gain: F::one(),
            },
        ))
}

/// Bandpass filter.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Input 2: Q
/// - Output 0: filtered audio
#[inline]
pub fn bandpass<T: Float, F: Real>() -> An<Svf<T, F, BandpassMode<F>>> {
    An(Svf::new(
        BandpassMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: F::new(440),
            q: F::one(),
            gain: F::one(),
        },
    ))
}

/// Bandpass filter centered at `f` Hz with Q value `q`.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn bandpass_hz<T: Float, F: Real>(f: T, q: T) -> An<FixedSvf<T, F, BandpassMode<F>>> {
    An(FixedSvf::new(
        BandpassMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: convert(f),
            q: convert(q),
            gain: F::one(),
        },
    ))
}

/// Bandpass filter with Q value `q`.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Output 0: filtered audio
#[inline]
pub fn bandpass_q<T: Float, F: Real>(
    q: T,
) -> An<Pipe<T, Stack<T, MultiPass<U2, T>, Constant<U1, T>>, Svf<T, F, BandpassMode<F>>>> {
    (multipass::<U2, T>() | dc(q))
        >> An(Svf::new(
            BandpassMode::default(),
            &SvfParams {
                sample_rate: convert(DEFAULT_SR),
                cutoff: F::new(440),
                q: convert(q),
                gain: F::one(),
            },
        ))
}

/// Notch filter.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Input 2: Q
/// - Output 0: filtered audio
#[inline]
pub fn notch<T: Float, F: Real>() -> An<Svf<T, F, NotchMode<F>>> {
    An(Svf::new(
        NotchMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: F::new(440),
            q: F::one(),
            gain: F::one(),
        },
    ))
}

/// Notch filter centered at `f` Hz with Q value `q`.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn notch_hz<T: Float, F: Real>(f: T, q: T) -> An<FixedSvf<T, F, NotchMode<F>>> {
    An(FixedSvf::new(
        NotchMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: convert(f),
            q: convert(q),
            gain: F::one(),
        },
    ))
}

/// Notch filter with Q value `q`.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Output 0: filtered audio
#[inline]
pub fn notch_q<T: Float, F: Real>(
    q: T,
) -> An<Pipe<T, Stack<T, MultiPass<U2, T>, Constant<U1, T>>, Svf<T, F, NotchMode<F>>>> {
    (multipass::<U2, T>() | dc(q))
        >> An(Svf::new(
            NotchMode::default(),
            &SvfParams {
                sample_rate: convert(DEFAULT_SR),
                cutoff: F::new(440),
                q: convert(q),
                gain: F::one(),
            },
        ))
}

/// Peaking filter.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Input 2: Q
/// - Output 0: filtered audio
#[inline]
pub fn peak<T: Float, F: Real>() -> An<Svf<T, F, PeakMode<F>>> {
    An(Svf::new(
        PeakMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: F::new(440),
            q: F::one(),
            gain: F::one(),
        },
    ))
}

/// Peaking filter centered at `f` Hz with Q value `q`.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn peak_hz<T: Float, F: Real>(f: T, q: T) -> An<FixedSvf<T, F, PeakMode<F>>> {
    An(FixedSvf::new(
        PeakMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: convert(f),
            q: convert(q),
            gain: F::one(),
        },
    ))
}

/// Peaking filter with Q value `q`.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Output 0: filtered audio
#[inline]
pub fn peak_q<T: Float, F: Real>(
    q: T,
) -> An<Pipe<T, Stack<T, MultiPass<U2, T>, Constant<U1, T>>, Svf<T, F, PeakMode<F>>>> {
    (multipass::<U2, T>() | dc(q))
        >> An(Svf::new(
            PeakMode::default(),
            &SvfParams {
                sample_rate: convert(DEFAULT_SR),
                cutoff: F::new(440),
                q: convert(q),
                gain: F::one(),
            },
        ))
}

/// Allpass filter.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Input 2: Q
/// - Output 0: filtered audio
#[inline]
pub fn allpass<T: Float, F: Real>() -> An<Svf<T, F, AllpassMode<F>>> {
    An(Svf::new(
        AllpassMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: F::new(440),
            q: F::one(),
            gain: F::one(),
        },
    ))
}

/// Allpass filter centered at `f` Hz with Q value `q`.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn allpass_hz<T: Float, F: Real>(f: T, q: T) -> An<FixedSvf<T, F, AllpassMode<F>>> {
    An(FixedSvf::new(
        AllpassMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: convert(f),
            q: convert(q),
            gain: F::one(),
        },
    ))
}

/// Allpass filter with Q value `q`.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Output 0: filtered audio
#[inline]
pub fn allpass_q<T: Float, F: Real>(
    q: T,
) -> An<Pipe<T, Stack<T, MultiPass<U2, T>, Constant<U1, T>>, Svf<T, F, AllpassMode<F>>>> {
    (multipass::<U2, T>() | dc(q))
        >> An(Svf::new(
            AllpassMode::default(),
            &SvfParams {
                sample_rate: convert(DEFAULT_SR),
                cutoff: F::new(440),
                q: convert(q),
                gain: F::one(),
            },
        ))
}

/// Bell filter with adjustable gain.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Input 2: Q
/// - Input 3: amplitude gain
/// - Output 0: filtered audio
#[inline]
pub fn bell<T: Float, F: Real>() -> An<Svf<T, F, BellMode<F>>> {
    An(Svf::new(
        BellMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: F::new(440),
            q: F::one(),
            gain: F::one(),
        },
    ))
}

/// Bell filter centered at `f` Hz with Q value `q` and amplitude gain `gain`.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn bell_hz<T: Float, F: Real>(f: T, q: T, gain: T) -> An<FixedSvf<T, F, BellMode<F>>> {
    An(FixedSvf::new(
        BellMode::default(),
        &SvfParams::<F> {
            sample_rate: convert(DEFAULT_SR),
            cutoff: convert(f),
            q: convert(q),
            gain: convert(gain),
        },
    ))
}

/// Bell filter with with Q value `q` and amplitude gain `gain`.
/// - Input 0: audio
/// - Input 1: center frequency
/// - Output 0: filtered audio
#[inline]
pub fn bell_q<T: Float, F: Real>(
    q: T,
    gain: T,
) -> An<Pipe<T, Stack<T, MultiPass<U2, T>, Constant<U2, T>>, Svf<T, F, BellMode<F>>>> {
    (multipass::<U2, T>() | dc((q, gain)))
        >> An(Svf::new(
            BellMode::default(),
            &SvfParams {
                sample_rate: convert(DEFAULT_SR),
                cutoff: F::new(440),
                q: convert(q),
                gain: convert(gain),
            },
        ))
}

/// Low shelf filter with adjustable gain.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Input 2: Q
/// - Input 3: amplitude gain
/// - Output 0: filtered audio
#[inline]
pub fn lowshelf<T: Float, F: Real>() -> An<Svf<T, F, LowshelfMode<F>>> {
    An(Svf::new(
        LowshelfMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: F::new(440),
            q: F::one(),
            gain: F::one(),
        },
    ))
}

/// Low shelf filter centered at `f` Hz with Q value `q` and amplitude gain `gain`.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn lowshelf_hz<T: Float, F: Real>(f: T, q: T, gain: T) -> An<FixedSvf<T, F, LowshelfMode<F>>> {
    An(FixedSvf::new(
        LowshelfMode::default(),
        &SvfParams::<F> {
            sample_rate: convert(DEFAULT_SR),
            cutoff: convert(f),
            q: convert(q),
            gain: convert(gain),
        },
    ))
}

/// Low shelf filter with with Q value `q` and amplitude gain `gain`.
/// - Input 0: audio
/// - Input 1: cutoff frequency
/// - Output 0: filtered audio
#[inline]
pub fn lowshelf_q<T: Float, F: Real>(
    q: T,
    gain: T,
) -> An<Pipe<T, Stack<T, MultiPass<U2, T>, Constant<U2, T>>, Svf<T, F, LowshelfMode<F>>>> {
    (multipass::<U2, T>() | dc((q, gain)))
        >> An(Svf::new(
            LowshelfMode::default(),
            &SvfParams {
                sample_rate: convert(DEFAULT_SR),
                cutoff: convert(440.0),
                q: convert(q),
                gain: F::one(),
            },
        ))
}

/// High shelf filter with adjustable gain.
/// - Input 0: audio
/// - Input 1: center frequency (Hz)
/// - Input 2: Q
/// - Input 3: amplitude gain
/// - Output 0: filtered audio
#[inline]
pub fn highshelf<T: Float, F: Real>() -> An<Svf<T, F, HighshelfMode<F>>> {
    An(Svf::new(
        HighshelfMode::default(),
        &SvfParams {
            sample_rate: convert(DEFAULT_SR),
            cutoff: F::new(440),
            q: F::one(),
            gain: F::one(),
        },
    ))
}

/// High shelf filter centered at `f` Hz with Q value `q` and amplitude gain `gain`.
/// - Input 0: audio
/// - Output 0: filtered audio
#[inline]
pub fn highshelf_hz<T: Float, F: Real>(
    f: T,
    q: T,
    gain: T,
) -> An<FixedSvf<T, F, HighshelfMode<F>>> {
    An(FixedSvf::new(
        HighshelfMode::default(),
        &SvfParams::<F> {
            sample_rate: convert(DEFAULT_SR),
            cutoff: convert(f),
            q: convert(q),
            gain: convert(gain),
        },
    ))
}

/// High shelf filter with with Q value `q` and amplitude gain `gain`.
/// - Input 0: audio
/// - Input 1: cutoff frequency
/// - Output 0: filtered audio
#[inline]
pub fn highshelf_q<T: Float, F: Real>(
    q: T,
    gain: T,
) -> An<Pipe<T, Stack<T, MultiPass<U2, T>, Constant<U2, T>>, Svf<T, F, HighshelfMode<F>>>> {
    (multipass::<U2, T>() | dc((q, gain)))
        >> An(Svf::new(
            HighshelfMode::default(),
            &SvfParams {
                sample_rate: convert(DEFAULT_SR),
                cutoff: convert(440.0),
                q: convert(q),
                gain: F::one(),
            },
        ))
}

/// Pulse wave oscillator.
/// - Input 0: frequency in Hz
/// - Input 1: pulse duty cycle in 0...1
/// - Output 0: pulse wave
pub struct PulseWave<T: Float> {
    pulse: An<
        Pipe<
            T,
            Pipe<
                T,
                Stack<T, WaveSynth<'static, T, U2>, Pass<T>>,
                Stack<
                    T,
                    Pass<T>,
                    Pipe<T, Binop<T, FrameAdd<U1, T>, Pass<T>, Pass<T>>, PhaseSynth<'static, T>>,
                >,
            >,
            Binop<T, FrameSub<U1, T>, Pass<T>, Pass<T>>,
        >,
    >,
}

#[allow(clippy::new_without_default)]
impl<T: Float> PulseWave<T> {
    pub fn new() -> Self {
        Self {
            pulse: (An(WaveSynth::<'static, T, U2>::new(DEFAULT_SR, &SAW_TABLE)) | pass())
                >> (pass()
                    | (pass() + pass())
                        >> An(PhaseSynth::<'static, T>::new(DEFAULT_SR, &SAW_TABLE)))
                >> pass() - pass(),
        }
    }
}

impl<T: Float> AudioNode for PulseWave<T> {
    const ID: u64 = 44;
    type Sample = T;
    type Inputs = U2;
    type Outputs = U1;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.pulse.reset(sample_rate);
    }
    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        self.pulse.tick(input)
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        self.pulse.process(size, input, output);
    }
    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        self.pulse.route(input, frequency)
    }
    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        self.pulse.ping(probe, hash).hash(Self::ID)
    }
}

/// Pulse wave oscillator.
/// - Input 0: frequency in Hz
/// - Input 1: pulse duty cycle in 0...1
/// - Output 0: pulse wave
#[inline]
pub fn pulse<T: Float>() -> An<PulseWave<T>> {
    An(PulseWave::new())
}

/// Morphing filter that morphs between lowpass, peak and highpass modes.
/// - Input 0: input signal
/// - Input 1: center frequency (Hz)
/// - Input 2: Q
/// - Input 3: morph in -1...1 (-1 = lowpass, 0 = peak, 1 = highpass)
/// - Output 0: filtered signal
pub struct Morph<T: Float, F: Real> {
    filter: Svf<T, F, PeakMode<F>>,
    morph: T,
}

impl<T: Float, F: Real> Morph<T, F> {
    pub fn new(sample_rate: f64, cutoff: F, q: F, morph: T) -> Self {
        let params = SvfParams {
            sample_rate: convert(sample_rate),
            cutoff,
            q,
            gain: F::zero(),
        };
        let mut node = Self {
            filter: Svf::new(PeakMode::new(), &params),
            morph,
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }
}

impl<T: Float, F: Real> AudioNode for Morph<T, F> {
    const ID: u64 = 62;
    type Sample = T;
    type Inputs = U4;
    type Outputs = U1;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.filter.reset(sample_rate);
    }
    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        self.morph = input[3];
        let filter_out = self.filter.tick(Frame::from_slice(&input[0..3]));
        [(filter_out[0] + input[3] * input[0]) * T::from_f32(0.5)].into()
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        self.filter.process(size, &input[0..3], output);
        for i in 0..size {
            output[0][i] = (output[0][i] + input[0][i] * input[3][i]) * T::from_f32(0.5);
        }
        self.morph = input[3][size - 1];
    }
    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        let mut output = self.filter.route(input, frequency);
        output[0] = output[0].filter(0.0, |r| {
            (r + Complex64::new(self.morph.to_f64(), 0.0)) * 0.5
        });
        output
    }
    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        self.filter.ping(probe, hash).hash(Self::ID)
    }
}

/// Morphing filter that morphs between lowpass, peak and highpass modes.
/// - Input 0: input signal
/// - Input 1: center frequency (Hz)
/// - Input 2: Q
/// - Input 3: morph in -1...1 (-1 = lowpass, 0 = peak, 1 = highpass)
/// - Output 0: filtered signal
pub fn morph<T: Real, F: Real>() -> An<Morph<T, F>> {
    An(Morph::new(DEFAULT_SR, F::new(440), F::one(), T::zero()))
}

/// Morphing filter with center frequency `f`, Q value `q`, and morph `morph`
/// (-1 = lowpass, 0 = peaking, 1 = highpass).
/// - Input 0: input signal
/// - Output 0: filtered signal
pub fn morph_hz<T: Real, F: Real>(
    f: T,
    q: T,
    morph: T,
) -> An<Pipe<T, Stack<T, Pass<T>, Constant<U3, T>>, Morph<T, F>>> {
    (pass() | dc((f, q, morph)))
        >> An(Morph::new(
            DEFAULT_SR,
            convert(f),
            convert(q),
            convert(morph),
        ))
}

/// Play back a channel of a Wave64.
/// Optional loop point is the index to jump to at the end of the wave.
/// - Output 0: wave
pub fn wave64<T: Float>(
    wave: Arc<Wave64>,
    channel: usize,
    loop_point: Option<usize>,
) -> An<Wave64Player<T>> {
    An(Wave64Player::new(wave, channel, loop_point))
}

/// Play back a channel of a Wave32.
/// Optional loop point is the index to jump to at the end of the wave.
/// - Output 0: wave
pub fn wave32<T: Float>(
    wave: Arc<Wave32>,
    channel: usize,
    loop_point: Option<usize>,
) -> An<Wave32Player<T>> {
    An(Wave32Player::new(wave, channel, loop_point))
}

/// Mono chorus, 5 voices. For stereo, stack two of these using different seed values.
/// `seed`: LFO seed.
/// `separation`: base voice separation in seconds (for example, 0.015).
/// `variation`: delay variation in seconds (for example, 0.005).
/// `mod_frequency`: delay modulation frequency (for example, 0.2).
/// - Input 0: audio.
/// - Output 0: chorused audio, including original signal.
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// chorus::<f64>(0, 0.015, 0.005, 0.5);
/// ```
pub fn chorus<T: Real>(
    seed: i64,
    separation: T,
    variation: T,
    mod_frequency: T,
) -> An<impl AudioNode<Sample = T, Inputs = U1, Outputs = U1>> {
    pass()
        & (pass()
            | lfo(move |t| {
                (
                    lerp11(
                        separation,
                        separation + variation,
                        spline_noise(seed, t * mod_frequency),
                    ),
                    lerp11(
                        separation * T::new(2),
                        separation * T::new(2) + variation,
                        spline_noise(hash(seed), t * (mod_frequency + T::from_f64(0.02))),
                    ),
                    lerp11(
                        separation * T::new(3),
                        separation * T::new(3) + variation,
                        spline_noise(
                            hash(seed ^ 0xfedcba),
                            t * (mod_frequency + T::from_f64(0.04)),
                        ),
                    ),
                    lerp11(
                        separation * T::new(4),
                        separation * T::new(4) + variation,
                        spline_noise(
                            hash(seed ^ 0xfedcb),
                            t * (mod_frequency + T::from_f64(0.06)),
                        ),
                    ),
                )
            }))
            >> multitap::<U4, T>(
                separation.to_f64(),
                (separation * T::new(4) + variation).to_f64(),
            )
}

/// Mono flanger.
/// `feedback_amount`: amount of feedback (for example, 0.9 or -0.9). Negative feedback inverts feedback phase.
/// `minimum_delay`: minimum delay in seconds (for example, 0.005).
/// `maximum_delay`: maximum delay in seconds (for example, 0.010).
/// ´delay_f´: Delay in `minimum_delay`...`maximum_delay` as a function of time. For example, `|t| lerp11(0.005, 0.010, sin_hz(0.1, t))`.
/// - Input 0: audio
/// - Output 0: flanged audio, including original signal
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// saw_hz(110.0) >> flanger::<f64, _>(0.9, 0.005, 0.010, |t| lerp11(0.005, 0.010, sin_hz(0.1, t)));
/// ```
pub fn flanger<T: Real, X: Fn(T) -> T>(
    feedback_amount: T,
    minimum_delay: T,
    maximum_delay: T,
    delay_f: X,
) -> An<impl AudioNode<Sample = T, Inputs = U1, Outputs = U1>> {
    pass()
        & feedback(
            (pass() | lfo(delay_f))
                >> tap::<T>(minimum_delay.to_f64(), maximum_delay.to_f64())
                >> shape(Shape::Tanh(feedback_amount)),
        )
}

/// Mono phaser.
/// `feedback_amount`: amount of feedback (for example, 0.5). Negative feedback inverts feedback phase.
/// `phase_f`: allpass modulation value in 0...1 as function of time, for example `|t| sin_hz(0.1, t) * 0.5 + 0.5`.
/// - Input 0: audio
/// - Output 0: phased audio
///
/// ### Example
/// ```
/// use fundsp::prelude::*;
/// saw_hz(110.0) >> phaser::<f64, _>(0.5, |t| sin_hz(0.1, t) * 0.5 + 0.5);
/// ```
pub fn phaser<T: Real, X: Fn(T) -> T>(
    feedback_amount: T,
    phase_f: X,
) -> An<impl AudioNode<Sample = T, Inputs = U1, Outputs = U1>> {
    pass()
        & feedback(
            (pass() | lfo(move |t| lerp(T::new(1), T::new(10), phase_f(t))))
                >> pipe::<U20, T, _, _>(|_i| {
                    (pass() | add(T::from_f64(0.05))) >> !allpole::<T, T>()
                })
                >> (mul(feedback_amount) | sink()),
        )
}
