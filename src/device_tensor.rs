// device_tensor.rs — Unified CPU/CUDA tensor dispatch for shrew-python

use shrew_core::dtype::DType;
use shrew_core::error::{Error, Result};
use shrew_core::shape::Shape;
use shrew_cpu::{CpuBackend, CpuDevice};

#[cfg(feature = "cuda")]
use shrew_cuda::{CudaBackend, CudaDevice};

/// Underlying tensor storage, either on host (CPU) or device (CUDA).
#[derive(Clone)]
pub enum DeviceTensor {
    Cpu(shrew_core::tensor::Tensor<CpuBackend>),
    #[cfg(feature = "cuda")]
    Cuda(shrew_core::tensor::Tensor<CudaBackend>),
}

macro_rules! match_unary {
    ($self:expr, $fn:ident) => {
        match $self {
            DeviceTensor::Cpu(t) => t.$fn().map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.$fn().map(DeviceTensor::Cuda),
        }
    };
}

macro_rules! match_binary {
    ($self:expr, $other:expr, $fn:ident) => {
        match ($self, $other) {
            (DeviceTensor::Cpu(a), DeviceTensor::Cpu(b)) => {
                a.$fn(b).map(DeviceTensor::Cpu)
            }
            #[cfg(feature = "cuda")]
            (DeviceTensor::Cuda(a), DeviceTensor::Cuda(b)) => {
                a.$fn(b).map(DeviceTensor::Cuda)
            }
            #[cfg(feature = "cuda")]
            (DeviceTensor::Cpu(a), DeviceTensor::Cuda(b)) => {
                let a_cuda = shrew_core::tensor::Tensor::<CudaBackend>::from_f64_slice(
                    &a.to_f64_vec()?,
                    a.shape().clone(),
                    a.dtype(),
                    b.device(),
                )?;
                a_cuda.$fn(b).map(DeviceTensor::Cuda)
            }
            #[cfg(feature = "cuda")]
            (DeviceTensor::Cuda(a), DeviceTensor::Cpu(b)) => {
                let b_cuda = shrew_core::tensor::Tensor::<CudaBackend>::from_f64_slice(
                    &b.to_f64_vec()?,
                    b.shape().clone(),
                    b.dtype(),
                    a.device(),
                )?;
                a.$fn(&b_cuda).map(DeviceTensor::Cuda)
            }
        }
    };
}

impl DeviceTensor {
    // ── Metadata ─────────────────────────────────────────────────────────────

    pub fn dims(&self) -> &[usize] {
        match self {
            DeviceTensor::Cpu(t) => t.dims(),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.dims(),
        }
    }

    pub fn shape(&self) -> &Shape {
        match self {
            DeviceTensor::Cpu(t) => t.shape(),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.shape(),
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            DeviceTensor::Cpu(t) => t.rank(),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.rank(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            DeviceTensor::Cpu(t) => t.dtype(),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.dtype(),
        }
    }

    pub fn elem_count(&self) -> usize {
        match self {
            DeviceTensor::Cpu(t) => t.elem_count(),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.elem_count(),
        }
    }

    pub fn is_variable(&self) -> bool {
        match self {
            DeviceTensor::Cpu(t) => t.is_variable(),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.is_variable(),
        }
    }

    pub fn is_contiguous(&self) -> bool {
        match self {
            DeviceTensor::Cpu(t) => t.is_contiguous(),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.is_contiguous(),
        }
    }

    pub fn is_cuda(&self) -> bool {
        match self {
            DeviceTensor::Cpu(_) => false,
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(_) => true,
        }
    }

    pub fn device_name(&self) -> String {
        match self {
            DeviceTensor::Cpu(_) => "cpu".to_string(),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => format!("cuda:{}", t.device().ordinal()),
        }
    }

    // ── Device Transfers ─────────────────────────────────────────────────────

    pub fn to_cpu(&self) -> Result<shrew_core::tensor::Tensor<CpuBackend>> {
        match self {
            DeviceTensor::Cpu(t) => Ok(t.clone()),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => {
                let data = t.to_f64_vec()?;
                shrew_core::tensor::Tensor::<CpuBackend>::from_f64_slice(
                    &data,
                    t.shape().clone(),
                    t.dtype(),
                    &CpuDevice,
                )
            }
        }
    }

    pub fn cpu(&self) -> Result<Self> {
        self.to_cpu().map(DeviceTensor::Cpu)
    }

    pub fn cuda(&self, _device_id: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            match self {
                DeviceTensor::Cuda(t) if t.device().ordinal() == _device_id => Ok(self.clone()),
                _ => {
                    let dev = CudaDevice::new(_device_id)?;
                    let data = self.to_f64_vec()?;
                    let t = shrew_core::tensor::Tensor::<CudaBackend>::from_f64_slice(
                        &data,
                        self.shape().clone(),
                        self.dtype(),
                        &dev,
                    )?;
                    Ok(DeviceTensor::Cuda(t))
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::msg("CUDA is not available in this build of Shrew"))
        }
    }

    pub fn to_f64_vec(&self) -> Result<Vec<f64>> {
        match self {
            DeviceTensor::Cpu(t) => t.to_f64_vec(),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.to_f64_vec(),
        }
    }

    pub fn to_scalar_f64(&self) -> Result<f64> {
        match self {
            DeviceTensor::Cpu(t) => t.to_scalar_f64(),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.to_scalar_f64(),
        }
    }

    pub fn to_dtype(&self, dt: DType) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.to_dtype(dt).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.to_dtype(dt).map(DeviceTensor::Cuda),
        }
    }

    // ── Creation Helpers ─────────────────────────────────────────────────────

    pub fn zeros(shape: &[usize], dtype: DType, device: &str) -> Result<Self> {
        if device == "cpu" {
            let t = shrew_core::tensor::Tensor::<CpuBackend>::zeros(shape, dtype, &CpuDevice)?;
            Ok(DeviceTensor::Cpu(t))
        } else if device.starts_with("cuda") {
            #[cfg(feature = "cuda")]
            {
                let ord = device.strip_prefix("cuda:").and_then(|s| s.parse().ok()).unwrap_or(0);
                let dev = CudaDevice::new(ord)?;
                let t = shrew_core::tensor::Tensor::<CudaBackend>::zeros(shape, dtype, &dev)?;
                Ok(DeviceTensor::Cuda(t))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(Error::msg("CUDA is not available in this build"))
            }
        } else {
            Err(Error::msg(format!("Unknown device: {device}")))
        }
    }

    pub fn ones(shape: &[usize], dtype: DType, device: &str) -> Result<Self> {
        if device == "cpu" {
            let t = shrew_core::tensor::Tensor::<CpuBackend>::ones(shape, dtype, &CpuDevice)?;
            Ok(DeviceTensor::Cpu(t))
        } else if device.starts_with("cuda") {
            #[cfg(feature = "cuda")]
            {
                let ord = device.strip_prefix("cuda:").and_then(|s| s.parse().ok()).unwrap_or(0);
                let dev = CudaDevice::new(ord)?;
                let t = shrew_core::tensor::Tensor::<CudaBackend>::ones(shape, dtype, &dev)?;
                Ok(DeviceTensor::Cuda(t))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(Error::msg("CUDA is not available in this build"))
            }
        } else {
            Err(Error::msg(format!("Unknown device: {device}")))
        }
    }

    pub fn full(shape: &[usize], val: f64, dtype: DType, device: &str) -> Result<Self> {
        if device == "cpu" {
            let t = shrew_core::tensor::Tensor::<CpuBackend>::full(shape, val, dtype, &CpuDevice)?;
            Ok(DeviceTensor::Cpu(t))
        } else if device.starts_with("cuda") {
            #[cfg(feature = "cuda")]
            {
                let ord = device.strip_prefix("cuda:").and_then(|s| s.parse().ok()).unwrap_or(0);
                let dev = CudaDevice::new(ord)?;
                let t = shrew_core::tensor::Tensor::<CudaBackend>::full(shape, val, dtype, &dev)?;
                Ok(DeviceTensor::Cuda(t))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(Error::msg("CUDA is not available in this build"))
            }
        } else {
            Err(Error::msg(format!("Unknown device: {device}")))
        }
    }

    pub fn rand(shape: &[usize], dtype: DType, device: &str) -> Result<Self> {
        if device == "cpu" {
            let t = shrew_core::tensor::Tensor::<CpuBackend>::rand(shape, dtype, &CpuDevice)?;
            Ok(DeviceTensor::Cpu(t))
        } else if device.starts_with("cuda") {
            #[cfg(feature = "cuda")]
            {
                let ord = device.strip_prefix("cuda:").and_then(|s| s.parse().ok()).unwrap_or(0);
                let dev = CudaDevice::new(ord)?;
                let t = shrew_core::tensor::Tensor::<CudaBackend>::rand(shape, dtype, &dev)?;
                Ok(DeviceTensor::Cuda(t))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(Error::msg("CUDA is not available in this build"))
            }
        } else {
            Err(Error::msg(format!("Unknown device: {device}")))
        }
    }

    pub fn randn(shape: &[usize], dtype: DType, device: &str) -> Result<Self> {
        if device == "cpu" {
            let t = shrew_core::tensor::Tensor::<CpuBackend>::randn(shape, dtype, &CpuDevice)?;
            Ok(DeviceTensor::Cpu(t))
        } else if device.starts_with("cuda") {
            #[cfg(feature = "cuda")]
            {
                let ord = device.strip_prefix("cuda:").and_then(|s| s.parse().ok()).unwrap_or(0);
                let dev = CudaDevice::new(ord)?;
                let t = shrew_core::tensor::Tensor::<CudaBackend>::randn(shape, dtype, &dev)?;
                Ok(DeviceTensor::Cuda(t))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(Error::msg("CUDA is not available in this build"))
            }
        } else {
            Err(Error::msg(format!("Unknown device: {device}")))
        }
    }

    pub fn from_f64_slice(data: &[f64], shape: &[usize], dtype: DType, device: &str) -> Result<Self> {
        if device == "cpu" {
            let t = shrew_core::tensor::Tensor::<CpuBackend>::from_f64_slice(data, shape, dtype, &CpuDevice)?;
            Ok(DeviceTensor::Cpu(t))
        } else if device.starts_with("cuda") {
            #[cfg(feature = "cuda")]
            {
                let ord = device.strip_prefix("cuda:").and_then(|s| s.parse().ok()).unwrap_or(0);
                let dev = CudaDevice::new(ord)?;
                let t = shrew_core::tensor::Tensor::<CudaBackend>::from_f64_slice(data, shape, dtype, &dev)?;
                Ok(DeviceTensor::Cuda(t))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(Error::msg("CUDA is not available in this build"))
            }
        } else {
            Err(Error::msg(format!("Unknown device: {device}")))
        }
    }

    pub fn zeros_like(&self) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => shrew_core::tensor::Tensor::<CpuBackend>::zeros_like(t).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => shrew_core::tensor::Tensor::<CudaBackend>::zeros_like(t).map(DeviceTensor::Cuda),
        }
    }

    pub fn ones_like(&self) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => shrew_core::tensor::Tensor::<CpuBackend>::ones_like(t).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => shrew_core::tensor::Tensor::<CudaBackend>::ones_like(t).map(DeviceTensor::Cuda),
        }
    }

    pub fn full_like(&self, val: f64) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => shrew_core::tensor::Tensor::<CpuBackend>::full_like(t, val).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => shrew_core::tensor::Tensor::<CudaBackend>::full_like(t, val).map(DeviceTensor::Cuda),
        }
    }

    // ── Unary Operations ─────────────────────────────────────────────────────

    pub fn relu(&self) -> Result<Self> { match_unary!(self, relu) }
    pub fn sigmoid(&self) -> Result<Self> { match_unary!(self, sigmoid) }
    pub fn tanh(&self) -> Result<Self> { match_unary!(self, tanh) }
    pub fn gelu(&self) -> Result<Self> { match_unary!(self, gelu) }
    pub fn silu(&self) -> Result<Self> { match_unary!(self, silu) }
    pub fn exp(&self) -> Result<Self> { match_unary!(self, exp) }
    pub fn log(&self) -> Result<Self> { match_unary!(self, log) }
    pub fn sqrt(&self) -> Result<Self> { match_unary!(self, sqrt) }
    pub fn neg(&self) -> Result<Self> { match_unary!(self, neg) }
    pub fn abs(&self) -> Result<Self> { match_unary!(self, abs) }
    pub fn sign(&self) -> Result<Self> { match_unary!(self, sign) }
    pub fn sin(&self) -> Result<Self> { match_unary!(self, sin) }
    pub fn cos(&self) -> Result<Self> { match_unary!(self, cos) }
    pub fn square(&self) -> Result<Self> { match_unary!(self, square) }
    pub fn floor(&self) -> Result<Self> { match_unary!(self, floor) }
    pub fn ceil(&self) -> Result<Self> { match_unary!(self, ceil) }
    pub fn round(&self) -> Result<Self> { match_unary!(self, round) }
    pub fn reciprocal(&self) -> Result<Self> { match_unary!(self, reciprocal) }
    pub fn rsqrt(&self) -> Result<Self> { match_unary!(self, rsqrt) }
    pub fn cumsum(&self, dim: usize) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.cumsum(dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.cumsum(dim).map(DeviceTensor::Cuda),
        }
    }
    pub fn contiguous(&self) -> Result<Self> { match_unary!(self, contiguous) }
    pub fn detach(&self) -> Self {
        match self {
            DeviceTensor::Cpu(t) => DeviceTensor::Cpu(t.detach()),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => DeviceTensor::Cuda(t.detach()),
        }
    }
    pub fn set_variable(&self) -> Self {
        match self {
            DeviceTensor::Cpu(t) => DeviceTensor::Cpu(t.clone().set_variable()),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => DeviceTensor::Cuda(t.clone().set_variable()),
        }
    }
    pub fn freeze(&self) -> Self {
        match self {
            DeviceTensor::Cpu(t) => DeviceTensor::Cpu(t.clone().freeze()),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => DeviceTensor::Cuda(t.clone().freeze()),
        }
    }
    pub fn unfreeze(&self) -> Self {
        match self {
            DeviceTensor::Cpu(t) => DeviceTensor::Cpu(t.clone().unfreeze()),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => DeviceTensor::Cuda(t.clone().unfreeze()),
        }
    }

    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.affine(mul, add).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.affine(mul, add).map(DeviceTensor::Cuda),
        }
    }

    pub fn clamp(&self, min: f64, max: f64) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.clamp(min, max).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.clamp(min, max).map(DeviceTensor::Cuda),
        }
    }

    pub fn powf(&self, exp: f64) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.powf(exp).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.powf(exp).map(DeviceTensor::Cuda),
        }
    }

    pub fn softmax(&self, dim: usize) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.softmax(dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.softmax(dim).map(DeviceTensor::Cuda),
        }
    }

    pub fn log_softmax(&self, dim: usize) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.log_softmax(dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.log_softmax(dim).map(DeviceTensor::Cuda),
        }
    }

    // ── Binary Operations ────────────────────────────────────────────────────

    pub fn add(&self, other: &Self) -> Result<Self> { match_binary!(self, other, add) }
    pub fn sub(&self, other: &Self) -> Result<Self> { match_binary!(self, other, sub) }
    pub fn mul(&self, other: &Self) -> Result<Self> { match_binary!(self, other, mul) }
    pub fn div(&self, other: &Self) -> Result<Self> { match_binary!(self, other, div) }
    pub fn matmul(&self, other: &Self) -> Result<Self> { match_binary!(self, other, matmul) }

    pub fn eq(&self, other: &Self) -> Result<Self> { match_binary!(self, other, eq) }
    pub fn ne(&self, other: &Self) -> Result<Self> { match_binary!(self, other, ne) }
    pub fn gt(&self, other: &Self) -> Result<Self> { match_binary!(self, other, gt) }
    pub fn ge(&self, other: &Self) -> Result<Self> { match_binary!(self, other, ge) }
    pub fn lt(&self, other: &Self) -> Result<Self> { match_binary!(self, other, lt) }
    pub fn le(&self, other: &Self) -> Result<Self> { match_binary!(self, other, le) }

    // ── Reductions ───────────────────────────────────────────────────────────

    pub fn sum_all(&self) -> Result<Self> { match_unary!(self, sum_all) }
    pub fn mean_all(&self) -> Result<Self> { match_unary!(self, mean_all) }

    pub fn sum(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.sum(dim, keep_dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.sum(dim, keep_dim).map(DeviceTensor::Cuda),
        }
    }

    pub fn mean(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.mean(dim, keep_dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.mean(dim, keep_dim).map(DeviceTensor::Cuda),
        }
    }

    pub fn max(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.max(dim, keep_dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.max(dim, keep_dim).map(DeviceTensor::Cuda),
        }
    }

    pub fn min(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.min(dim, keep_dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.min(dim, keep_dim).map(DeviceTensor::Cuda),
        }
    }

    pub fn var(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.var(dim, keep_dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.var(dim, keep_dim).map(DeviceTensor::Cuda),
        }
    }

    pub fn std(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.std(dim, keep_dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.std(dim, keep_dim).map(DeviceTensor::Cuda),
        }
    }

    pub fn logsumexp(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.logsumexp(dim, keep_dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.logsumexp(dim, keep_dim).map(DeviceTensor::Cuda),
        }
    }

    pub fn prod(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.prod(dim, keep_dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.prod(dim, keep_dim).map(DeviceTensor::Cuda),
        }
    }

    pub fn masked_fill(&self, mask: &Self, value: f64) -> Result<Self> {
        match (self, mask) {
            (DeviceTensor::Cpu(t), DeviceTensor::Cpu(m)) => {
                t.masked_fill(m, value).map(DeviceTensor::Cpu)
            }
            #[cfg(feature = "cuda")]
            (DeviceTensor::Cuda(t), DeviceTensor::Cuda(m)) => {
                t.masked_fill(m, value).map(DeviceTensor::Cuda)
            }
            #[cfg(feature = "cuda")]
            _ => {
                let cpu_t = self.to_cpu()?;
                let cpu_m = mask.to_cpu()?;
                cpu_t.masked_fill(&cpu_m, value).map(DeviceTensor::Cpu)
            }
        }
    }

    pub fn argmax(&self, dim: usize, keepdim: bool) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.argmax(dim, keepdim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.argmax(dim, keepdim).map(DeviceTensor::Cuda),
        }
    }

    pub fn argmin(&self, dim: usize, keepdim: bool) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.argmin(dim, keepdim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.argmin(dim, keepdim).map(DeviceTensor::Cuda),
        }
    }

    // ── Shape Operations ─────────────────────────────────────────────────────

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.transpose(dim0, dim1).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.transpose(dim0, dim1).map(DeviceTensor::Cuda),
        }
    }

    pub fn t(&self) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.t().map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.t().map(DeviceTensor::Cuda),
        }
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.permute(dims).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.permute(dims).map(DeviceTensor::Cuda),
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.reshape(shape).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.reshape(shape).map(DeviceTensor::Cuda),
        }
    }

    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.squeeze(dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.squeeze(dim).map(DeviceTensor::Cuda),
        }
    }

    pub fn squeeze_all(&self) -> Self {
        match self {
            DeviceTensor::Cpu(t) => DeviceTensor::Cpu(t.squeeze_all()),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => DeviceTensor::Cuda(t.squeeze_all()),
        }
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.unsqueeze(dim).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.unsqueeze(dim).map(DeviceTensor::Cuda),
        }
    }

    pub fn flatten(&self, start: usize, end: usize) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.flatten(start, end).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.flatten(start, end).map(DeviceTensor::Cuda),
        }
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.narrow(dim, start, len).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.narrow(dim, start, len).map(DeviceTensor::Cuda),
        }
    }

    pub fn expand(&self, shape: &[usize]) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.expand(shape).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.expand(shape).map(DeviceTensor::Cuda),
        }
    }

    pub fn chunk(&self, n: usize, dim: usize) -> Result<Vec<Self>> {
        match self {
            DeviceTensor::Cpu(t) => Ok(t.chunk(n, dim)?.into_iter().map(DeviceTensor::Cpu).collect()),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => Ok(t.chunk(n, dim)?.into_iter().map(DeviceTensor::Cuda).collect()),
        }
    }

    pub fn split(&self, split_size: usize, dim: usize) -> Result<Vec<Self>> {
        match self {
            DeviceTensor::Cpu(t) => Ok(t.split(split_size, dim)?.into_iter().map(DeviceTensor::Cpu).collect()),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => Ok(t.split(split_size, dim)?.into_iter().map(DeviceTensor::Cuda).collect()),
        }
    }

    pub fn pad(&self, padding: &[[usize; 2]], value: f64) -> Result<Self> {
        match self {
            DeviceTensor::Cpu(t) => t.pad(padding, value).map(DeviceTensor::Cpu),
            #[cfg(feature = "cuda")]
            DeviceTensor::Cuda(t) => t.pad(padding, value).map(DeviceTensor::Cuda),
        }
    }

    pub fn conv2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Self> {
        match (self, weight) {
            (DeviceTensor::Cpu(x), DeviceTensor::Cpu(w)) => {
                #[allow(unreachable_patterns)]
                let b = bias.and_then(|b| match b {
                    DeviceTensor::Cpu(bt) => Some(bt),
                    _ => None,
                });
                x.conv2d(w, b, stride, padding).map(DeviceTensor::Cpu)
            }
            #[cfg(feature = "cuda")]
            (DeviceTensor::Cuda(x), DeviceTensor::Cuda(w)) => {
                let b_cuda = if let Some(b) = bias {
                    Some(b.to_cuda_internal(x.device())?)
                } else {
                    None
                };
                x.conv2d(w, b_cuda.as_ref(), stride, padding).map(DeviceTensor::Cuda)
            }
            #[cfg(feature = "cuda")]
            _ => {
                let cpu_x = self.to_cpu()?;
                let cpu_w = weight.to_cpu()?;
                let cpu_b = if let Some(b) = bias { Some(b.to_cpu()?) } else { None };
                cpu_x.conv2d(&cpu_w, cpu_b.as_ref(), stride, padding).map(DeviceTensor::Cpu)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn to_cuda_internal(&self, dev: &CudaDevice) -> Result<shrew_core::tensor::Tensor<CudaBackend>> {
        match self {
            DeviceTensor::Cuda(t) => Ok(t.clone()),
            DeviceTensor::Cpu(t) => {
                let data = t.to_f64_vec()?;
                shrew_core::tensor::Tensor::<CudaBackend>::from_f64_slice(&data, t.shape().clone(), t.dtype(), dev)
            }
        }
    }
}
