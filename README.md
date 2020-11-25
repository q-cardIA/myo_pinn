# Physics-informed neural networks for myocardial perfusion MRI quantification
Rudolf L.M. van Herten, Amedeo Chiribiri, Marcel Breeuwer, Mitko Veta, and Cian M. Scannell.

# Abstract 
Tracer-kinetic models allow for the quantification of kinetic parameters such as blood flow from dynamic contrast-enhanced magnetic resonance (MR) images. Fitting the observed data with multi-compartment exchange models is desirable, as they are physiologically plausible and resolve directly for blood flow and microvascular function. However, the reliability of model fitting is limited by the low signal-to-noise ratio, temporal resolution, and acquisition length. This may result in inaccurate parameter estimates.

This study introduces physics-informed neural networks (PINNs) as a means to perform myocardial perfusion MR quantification, which provides a versatile scheme for the inference of kinetic parameters. These neural networks can be trained to fit the observed perfusion MR data while respecting the underlying physical conservation laws described by a multi-compartment exchange model. Here, we provide a framework for the implementation of PINNs in myocardial perfusion MR.

The approach is validated both *in silico* and *in vivo*. In the *in silico* study, an overall reduction in mean-squared error with the ground-truth parameters was observed compared to a standard non-linear least squares fitting approach. The *in vivo* study demonstrates that the method produces parameter values comparable to those previously found in literature, as well as providing parameter maps which match the clinical diagnosis of patients.
