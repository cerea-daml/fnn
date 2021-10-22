
!> @brief Module dedicated to the class \ref tanhactivation.
module fnn_activation_tanh

    use fnn_common
    use fnn_activation_nonlinear

    implicit none

    private
    public :: TanhActivation, construct_tanh_activation

    !--------------------------------------------------
    !> @brief Implements a `tanh` activation function.
    type, extends(NonLinearActivation) :: TanhActivation
    contains
        !> @brief Saves the activation function.
        !> @details Implemented by \ref tanh_tofile.
        procedure, pass, public :: tofile => tanh_tofile
        !> @brief Applies and linearises the activation function.
        !> @details Implemented by \ref tanh_apply_forward.
        procedure, pass, public :: apply_forward => tanh_apply_forward
    end type TanhActivation

contains

    !--------------------------------------------------
    !> @brief Constructor for class \ref tanhactivation.
    !> @param[in] self_size The value for \ref linearactivation::self_size.
    !> @param[in] batch_size The value for \ref linearactivation::batch_size.
    !> @return The constructed activation function.
    type(TanhActivation) function construct_tanh_activation(self_size, batch_size) result(self)
        integer(ik), intent(in) :: self_size
        integer(ik), intent(in) :: batch_size
        self % NonLinearActivation = construct_nonlinear_activation(self_size, batch_size)
    end function construct_tanh_activation

    !--------------------------------------------------
    !> @brief Implements \ref tanhactivation::tofile.
    !>
    !> Saves the activation function.
    !> @param[in] self The activation function to save.
    !> @param[in] unit_num The unit number for the write statement.
    subroutine tanh_tofile(self, unit_num)
        class(TanhActivation), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        write(unit_num, fmt=*) 'tanh'
    end subroutine tanh_tofile

    !--------------------------------------------------
    !> @brief Implements \ref tanhactivation::apply_forward.
    !>
    !> Applies and linearises the activation function.
    !> @details The activation function reads
    !> \f[ \mathbf{y} = \mathcal{A}(\mathbf{z}) = \mathrm{tanh}(\mathbf{z}),\f]
    !> and the associated linearisation reads
    !> \f[ \mathbf{A}(\mathbf{z}) = \mathrm{diag}(1-\mathrm{tanh}(\mathbf{z})^2).\f]
    !>
    !> \b Note
    !>
    !> Input parameter `member` should be less than \ref linearactivation::batch_size.
    !>
    !> The linarisation is stored in nonlinearactivation::z_prime,
    !> which is why the intent of `self` is declared `inout`.
    !> @param[inout] self The activation function.
    !> @param[in] member The index inside the batch.
    !> @param[in] z The input of the activation function.
    !> @param[out] y The output of the activation function.
    subroutine tanh_apply_forward(self, member, z, y)
        class(TanhActivation), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: z(:)
        real(rk), intent(out) :: y(:)
        y = tanh(z)
        self % z_prime(:, member) = 1 - y**2
    end subroutine tanh_apply_forward

end module fnn_activation_tanh

