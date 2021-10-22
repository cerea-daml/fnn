
!> @brief Module dedicated to the class \ref nonlinearactivation.
module fnn_activation_nonlinear

    use fnn_common
    use fnn_activation_linear

    implicit none

    private
    public :: NonLinearActivation, construct_nonlinear_activation

    !--------------------------------------------------
    !> @brief Base class for nonlinear activation functions.
    !> Do not instanciate.
    !> @details This class only exists to implement the features
    !> which are common to all nonlinear activation functions:
    !> - the storage for the linearisation;
    !> - the TL of the activation function;
    !> - the adjoint of the activation function.
    type, extends(LinearActivation) :: NonLinearActivation
        private
        !> @brief The storage for the linearisation.
        !> @details Array of reals with shape (linearactivation::self_size, linearactivation::batch_size).
        real(rk), allocatable, public :: z_prime(:, :)
    contains
        !> @brief Applies the TL of the activation function.
        !> @details Implemented by \ref nonlinear_apply_tangent_linear.
        procedure, pass, public :: apply_tangent_linear => nonlinear_apply_tangent_linear
        !> @brief Applies the adjoint of the activation function.
        !> @details Implemented by \ref nonlinear_apply_adjoint.
        procedure, pass, public :: apply_adjoint => nonlinear_apply_adjoint
    end type NonLinearActivation

contains

    !--------------------------------------------------
    !> @brief Constructor for class \ref nonlinearactivation.
    !> @param[in] self_size The value for \ref linearactivation::self_size.
    !> @param[in] batch_size The value for \ref linearactivation::batch_size.
    !> @return The constructed activation function.
    type(NonLinearActivation) function construct_nonlinear_activation(self_size, batch_size) result(self)
        integer(ik), intent(in) :: self_size
        integer(ik), intent(in) :: batch_size
        self % LinearActivation = construct_linear_activation(self_size, batch_size)
        allocate(self % z_prime(self_size, batch_size))
    end function construct_nonlinear_activation

    !--------------------------------------------------
    !> @brief Implements \ref nonlinearactivation::apply_tangent_linear.
    !>
    !> Applies the TL of the activation function.
    !> @details The TL operator reads
    !> \f[ d\mathbf{y} = \mathbf{A}(\mathbf{z}) d\mathbf{z}.\f]
    !>
    !> \b Note
    !>
    !> In principle, this method should only be called after
    !> \ref linearactivation::apply_forward, where the linearisation 
    !> is computed.
    !> @param[in] self The activation function.
    !> @param[in] member The index inside the batch.
    !> @param[in] dz The input of the TL operator.
    !> @param[out] dy The output of the TL operator.
    subroutine nonlinear_apply_tangent_linear(self, member, dz, dy)
        class(NonLinearActivation), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dz(:)
        real(rk), intent(out) :: dy(:)
        dy = self % z_prime(:, member) * dz
    end subroutine nonlinear_apply_tangent_linear

    !--------------------------------------------------
    !> @brief Implements \ref nonlinearactivation::apply_adjoint.
    !>
    !> Applies the adjoint of the activation function.
    !> @details The adjoint operator reads
    !> \f[ d\mathbf{z} = \mathbf{A}(\mathbf{z})^\top d\mathbf{y}.\f]
    !>
    !> \b Note
    !>
    !> In principle, this method should only be called after
    !> \ref linearactivation::apply_forward, where the 
    !> linearisation is computed.
    !> @param[in] self The activation function.
    !> @param[in] member The index inside the batch.
    !> @param[in] dy The input of the adjoint operator.
    !> @param[out] dz The output of the adjoint operator.
    subroutine nonlinear_apply_adjoint(self, member, dy, dz)
        class(NonLinearActivation), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dy(:)
        real(rk), intent(out) :: dz(:)
        dz = self % z_prime(:, member) * dy
    end subroutine nonlinear_apply_adjoint

end module fnn_activation_nonlinear

