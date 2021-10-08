
module mod_activation

    use mod_kinds, only: ik, rk

    implicit none

    private
    public :: LinearActivation, construct_linear_activation, TanhActivation, construct_tanh_activation

    !--------------------------------------------------
    ! class LinearActivation
    !--------------------------------------------------
    !
    ! Base class for all activation functions.
    ! Implements a linear activation function.
    !
    ! Attributes
    ! ----------
    ! self_size : int
    !     The dimension of the input and output of the function.
    ! ensemble_size : int
    !     The number of simultaneous linearisations to compute.
    !

    type :: LinearActivation
        private
        integer(ik) :: self_size
        integer(ik) :: ensemble_size
    contains
        procedure, public, pass :: tofile => linear_tofile
        procedure, public, pass :: apply_activation => linear_apply_activation
        procedure, public, pass :: apply_tangent_linear => linear_apply_tangent_linear
        procedure, public, pass :: apply_adjoint => linear_apply_adjoint
    end type LinearActivation

    !--------------------------------------------------
    ! class NonLinearActivation
    !--------------------------------------------------
    !
    ! Extends LinearActivation.
    ! Base class for all nonlinear activation functions.
    ! Do not instanciate.
    !
    ! Attributes
    ! ----------
    ! z_prime : 2d array of reals
    !     The storage for the linearisations.
    !     Should have shape (self_size, ensemble_size).
    !

    type, extends(LinearActivation) :: NonLinearActivation
        private
        real(rk), allocatable :: z_prime(:, :)
    contains
        procedure, public, pass :: apply_tangent_linear => nonlinear_apply_tangent_linear
        procedure, public, pass :: apply_adjoint => nonlinear_apply_adjoint
    end type NonLinearActivation

    !--------------------------------------------------
    ! class TanhActivation
    !--------------------------------------------------
    !
    ! Extends NonLinearActivation.
    ! Implements a tanh activation function
    !

    type, extends(NonLinearActivation) :: TanhActivation
        private
    contains
        procedure, public, pass :: tofile => tanh_tofile
        procedure, public, pass :: apply_activation => tanh_apply_activation
    end type TanhActivation

contains

    !--------------------------------------------------
    ! methods for class LinearActivation
    !--------------------------------------------------

    type(LinearActivation) function construct_linear_activation(self_size, ensemble_size) result(self)
        ! Constructs a LinearActivation instance.
        !
        ! Parameters
        ! ----------
        ! self_size : int
        !     The value for activation % self_size.
        ! ensemble_size : int
        !     The value for activation % ensemble_size.
        !
        ! Returns
        ! -------
        ! activation : LinearActivation
        !     The constructed activation function.
        !     
        integer(ik), intent(in) :: self_size
        integer(ik), intent(in) :: ensemble_size
        !print *, 'CALLING construct_linear_activation()'
        self % self_size = self_size
        self % ensemble_size = ensemble_size
    end function construct_linear_activation

    subroutine linear_tofile(self, unit_num)
        class(LinearActivation), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        !print *, 'CALLING linear_tofile()'
        write(unit_num, fmt=*) 'linear'
    end subroutine linear_tofile

    subroutine linear_apply_activation(self, member, z, y)
        ! Applies the activation function:
        !     y = A(z) = z,
        ! and computes the associated linearisation dA(z) = I.
        !
        ! Parameters
        ! ----------
        ! member : int
        !     Index of the linearisation to compute.
        !     Should be less than ensemble_size.
        ! z : 1d array of real, size self_size
        !     Input of the activation function.
        ! y : 1d array of real, size self_size
        !     Output of the activation function.
        !
        ! Notes
        ! -----
        ! In this case, the linearisation is trivial and does not
        ! require any operation.
        !
        class(LinearActivation), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: z(:)
        real(rk), intent(out) :: y(:)
        !print *, 'CALLING linear_apply_activation()'
        y = z
    end subroutine linear_apply_activation

    subroutine linear_apply_tangent_linear(self, member, dz, dy)
        ! Applies the TL of the activation function:
        !     dy = dA(z) dz = dz.
        !
        ! Parameters
        ! ----------
        ! member : int
        !     Index of the linearisation to apply.
        !     Should be less than ensemble_size.
        ! dz : 1d array of real, size self_size
        !     Input of the TL operator.
        ! dy : 1d array of real, size self_size
        !     Output of the TL operator.
        !
        ! Notes
        ! -----
        ! In principle, this method should only be called after `apply_activation`,
        ! where the linearisation is computed. 
        !
        class(LinearActivation), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dz(:)
        real(rk), intent(out) :: dy(:)
        !print *, 'CALLING linear_apply_tangent_linear()'
        dy = dz
    end subroutine linear_apply_tangent_linear

    subroutine linear_apply_adjoint(self, member, dy, dz)
        ! Applies the adjoint of the activation function:
        !     dz = dA'(z) dy = dy.
        !
        ! Parameters
        ! ----------
        ! member : int
        !     Index of the linearisation to apply.
        !     Should be less than ensemble_size.
        ! dy : 1d array of real, size self_size
        !     Input of the adjoint operator.
        ! dz : 1d array of real, size self_size
        !     Output of the adjoint operator.
        !
        ! Notes
        ! -----
        ! In principle, this method should only be called after `apply_activation`,
        ! where the linearisation is computed. 
        !
        class(LinearActivation), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dy(:)
        real(rk), intent(out) :: dz(:)
        !print *, 'CALLING linear_apply_adjoint()'
        dz = dy
    end subroutine linear_apply_adjoint

    !--------------------------------------------------
    ! methods for class NonLinearActivation
    !--------------------------------------------------

    type(NonLinearActivation) function construct_nonlinear_activation(self_size, ensemble_size) result(self)
        ! Constructs a NonLinearActivation instance.
        !
        ! Parameters
        ! ----------
        ! self_size : int
        !     The value for activation % self_size.
        ! ensemble_size : int
        !     The value for activation % ensemble_size.
        !
        ! Returns
        ! -------
        ! activation : NonLinearActivation
        !     The constructed activation function.
        !     
        integer(ik), intent(in) :: self_size
        integer(ik), intent(in) :: ensemble_size
        !print *, 'CALLING construct_nonlinear_activation()'
        self % LinearActivation = construct_linear_activation(self_size, ensemble_size)
        allocate(self % z_prime(self_size, ensemble_size))
    end function construct_nonlinear_activation

    subroutine nonlinear_apply_tangent_linear(self, member, dz, dy)
        ! Applies the TL of the activation function:
        !     dy = dA(z) dz.
        !
        ! Parameters
        ! ----------
        ! member : int
        !     Index of the linearisation to apply.
        !     Should be less than ensemble_size.
        ! dz : 1d array of real, size self_size
        !     Input of the TL operator.
        ! dy : 1d array of real, size self_size
        !     Output of the TL operator.
        !
        ! Notes
        ! -----
        ! This method can only be called after `apply_activation`,
        ! where the linearisation is computed.
        !
        class(NonLinearActivation), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dz(:)
        real(rk), intent(out) :: dy(:)
        !print *, 'CALLING nonlinear_apply_tangent_linear()'
        dy = self % z_prime(:, member) * dz
    end subroutine nonlinear_apply_tangent_linear

    subroutine nonlinear_apply_adjoint(self, member, dy, dz)
        ! Applies the adjoint of the activation function:
        !     dz = dA'(z) dy.
        !
        ! Parameters
        ! ----------
        ! member : int
        !     Index of the linearisation to apply.
        !     Should be less than ensemble_size.
        ! dy : 1d array of real, size self_size
        !     Input of the adjoint operator.
        ! dz : 1d array of real, size self_size
        !     Output of the adjoint operator.
        !
        ! Notes
        ! -----
        ! This method can only be called after `apply_activation`,
        ! where the linearisation is computed.
        !
        class(NonLinearActivation), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dy(:)
        real(rk), intent(out) :: dz(:)
        !print *, 'CALLING nonlinear_apply_adjoint()'
        dz = self % z_prime(:, member) * dy
    end subroutine nonlinear_apply_adjoint

    !--------------------------------------------------
    ! methods for class TanhActivation
    !--------------------------------------------------

    type(TanhActivation) function construct_tanh_activation(self_size, ensemble_size) result(self)
        ! Constructs a TanhActivation instance.
        !
        ! Parameters
        ! ----------
        ! self_size : int
        !     The dimension of the input and output of the function.
        ! ensemble_size : int
        !     The value for activation % ensemble_size.
        !
        ! Returns
        ! -------
        ! activation : TanhActivation
        !     The constructed activation function.
        !     
        integer(ik), intent(in) :: self_size
        integer(ik), intent(in) :: ensemble_size
        !print *, 'CALLING construct_tanh_activation()'
        self % NonLinearActivation = construct_nonlinear_activation(self_size, ensemble_size)
    end function construct_tanh_activation

    subroutine tanh_tofile(self, unit_num)
        class(TanhActivation), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        !print *, 'CALLING tanh_tofile()'
        write(unit_num, fmt=*) 'tanh'
    end subroutine tanh_tofile

    subroutine tanh_apply_activation(self, member, z, y)
        ! Applies the activation function:
        !     y = A(z),
        ! and computes the associated linearisation dA(z).
        !
        ! Parameters
        ! ----------
        ! member : int
        !     Index of the linearisation to compute.
        !     Should be less than ensemble_size.
        ! z : 1d array of real, size self_size
        !     Input of the activation function.
        ! y : 1d array of real, size self_size
        !     Output of the activation function.
        !
        class(TanhActivation), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: z(:)
        real(rk), intent(out) :: y(:)
        !print *, 'CALLING tanh_apply_activation()'
        y = tanh(z)
        self % z_prime(:, member) = 1 - y**2
    end subroutine tanh_apply_activation

end module mod_activation

