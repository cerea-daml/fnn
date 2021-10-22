
!> @brief Module dedicated to the class \ref linearactivation.
module fnn_activation_linear

    use fnn_common

    implicit none

    private
    public :: LinearActivation, construct_linear_activation

    !--------------------------------------------------
    !> @brief Base class for all activation functions.
    !> Implements a linear activation function.
    type :: LinearActivation
        private
        !> The dimension of the input and output of the activation function.
        integer(ik) :: self_size
        !> The batch size.
        integer(ik) :: batch_size
    contains
        !> @brief Saves the activation function.
        !> @details Implemented by \ref linear_tofile.
        procedure, pass, public :: tofile => linear_tofile
        !> @brief Applies and linearises the activation function.
        !> @details Implemented by \ref linear_apply_forward.
        procedure, pass, public :: apply_forward => linear_apply_forward
        !> @brief Applies the TL of the activation function.
        !> @details Implemented by \ref linear_apply_tangent_linear.
        procedure, pass, public :: apply_tangent_linear => linear_apply_tangent_linear
        !> @brief Applies the adjoint of the activation function.
        !> @details Implemented by \ref linear_apply_adjoint.
        procedure, pass, public :: apply_adjoint => linear_apply_adjoint
    end type LinearActivation

contains

    !--------------------------------------------------
    !> @brief Constructor for class \ref linearactivation.
    !> @param[in] self_size The value for \ref linearactivation::self_size.
    !> @param[in] batch_size The value for \ref linearactivation::batch_size. 
    !> @return The constructed activation function.
    type(LinearActivation) function construct_linear_activation(self_size, batch_size) result(self)
        integer(ik), intent(in) :: self_size
        integer(ik), intent(in) :: batch_size
        self % self_size = self_size
        self % batch_size = batch_size
    end function construct_linear_activation

    !--------------------------------------------------
    !> @brief Implements \ref linearactivation::tofile.
    !>
    !> Saves the activation function.
    !> @param[in] self The activation function to save.
    !> @param[in] unit_num The unit number for the write statement.
    subroutine linear_tofile(self, unit_num)
        class(LinearActivation), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        write(unit_num, fmt=*) 'linear'
    end subroutine linear_tofile

    !--------------------------------------------------
    !> @brief Implements \ref linearactivation::apply_forward.
    !>
    !> Applies and linearises the activation function.
    !> @details The activation function reads
    !> \f[ \mathbf{y} = \mathcal{A}(\mathbf{z}) = \mathbf{z},\f]
    !> and the associated linearisation reads
    !> \f[ \mathbf{A}(\mathbf{z}) = \mathbf{I},\f]
    !> which is trivial and does not require any operation.
    !> 
    !> \b Note 
    !>
    !> Input parameter `member` should be less than \ref linearactivation::batch_size.
    !>
    !> This method supports inplace operations:
    !> it can be called with `y` = `z`.
    !>
    !> The intent for `self` is declared `inout` instead of `in` because,
    !> for certain subclasses (e.g. \ref fnn_activation_tanh::tanhactivation)
    !> the linearisation is stored inside the activation function.
    !> @param[inout] self The activation function.
    !> @param[in] member The index inside the batch.
    !> @param[in] z The input of the activation function.
    !> @param[out] y The output of the activation function.
    subroutine linear_apply_forward(self, member, z, y)
        class(LinearActivation), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: z(:)
        real(rk), intent(out) :: y(:)
        y = z
    end subroutine linear_apply_forward

    !--------------------------------------------------
    !> @brief Implements \ref linearactivation::apply_tangent_linear.
    !>
    !> Applies the TL of the activation function.
    !> @details The TL operator reads
    !> \f[ d\mathbf{y} = \mathbf{A}(\mathbf{z}) d\mathbf{z} = d\mathbf{z}.\f]
    !> 
    !> \b Note 
    !>
    !> In principle, this method should only be called after 
    !> \ref linearactivation::apply_forward,
    !> where the linearisation is computed.
    !>
    !> This method supports inplace operations:
    !> it can be called with `dy` = `dz`.
    !> @param[in] self The activation function.
    !> @param[in] member The index inside the batch.
    !> @param[in] dz The input of the TL operator.
    !> @param[out] dy The output of the TL operator.
    subroutine linear_apply_tangent_linear(self, member, dz, dy)
        class(LinearActivation), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dz(:)
        real(rk), intent(out) :: dy(:)
        dy = dz
    end subroutine linear_apply_tangent_linear

    !--------------------------------------------------
    !> @brief Implements \ref linearactivation::apply_adjoint.
    !>
    !> Applies the adjoint of the activation function.
    !> @details The adjoint operator reads
    !> \f[ d\mathbf{z} = \mathbf{A}(\mathbf{z})^\top d\mathbf{y} = d\mathbf{y}.\f]
    !> 
    !> \b Note 
    !>
    !> In principle, this method should only be called after 
    !> \ref linearactivation::apply_forward,
    !> where the linearisation is computed.
    !>
    !> This method supports inplace operations:
    !> it can be called with `dz` = `dy`.
    !> @param[in] self The activation function.
    !> @param[in] member The index inside the batch.
    !> @param[in] dy The input of the adjoint operator.
    !> @param[out] dz The output of the adjoint operator.
    subroutine linear_apply_adjoint(self, member, dy, dz)
        class(LinearActivation), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dy(:)
        real(rk), intent(out) :: dz(:)
        dz = dy
    end subroutine linear_apply_adjoint

end module fnn_activation_linear

