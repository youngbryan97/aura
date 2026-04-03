use pyo3::prelude::*;
use std::os::raw::c_int;

// Extern link to macOS-specific thread QoS APIs
extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: c_int) -> c_int;
}

// macOS QoS Class definitions (standard for Apple Silicon)
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21; // P-cores (Real-time, UI)
const QOS_CLASS_USER_INITIATED: u32 = 0x19;   // P-cores (Fast compute)
const QOS_CLASS_UTILITY: u32 = 0x15;          // E-cores (Background IO/Sensory)

#[pyfunction]
fn pin_to_p_cores() {
    unsafe {
        // Elevate current thread to User Initiated QoS (Apple Silicon P-Cores)
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    }
}

#[pyfunction]
fn pin_to_e_cores() {
    unsafe {
        // Set current thread to Utility QoS (Apple Silicon E-Cores for low power/IO)
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    }
}

// Apple Silicon NEON-accelerated dot product (Zero-copy)
#[pyfunction]
fn neon_dot_product(a: Vec<f32>, b: Vec<f32>) -> f32 {
    use core::arch::aarch64::*;
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    let mut i = 0;
    
    // Process in blocks of 4 using NEON intrinsics
    while i + 4 <= len {
        unsafe {
            let va = vld1q_f32(a[i..].as_ptr());
            let vb = vld1q_f32(b[i..].as_ptr());
            let prod = vmulq_f32(va, vb);
            sum += vaddvq_f32(prod); // Vector across-lane sum
        }
        i += 4;
    }
    
    // Scalar tail for remaining elements
    for j in i..len {
        sum += a[j] * b[j];
    }
    sum
}

#[pymodule]
fn aura_m1_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pin_to_p_cores, m)?)?;
    m.add_function(wrap_pyfunction!(pin_to_e_cores, m)?)?;
    m.add_function(wrap_pyfunction!(neon_dot_product, m)?)?;
    Ok(())
}
