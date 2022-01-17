
!> @brief Module dedicated to the class \ref reluactivation.
module fnn_activation_relu

    use fnn_common
    use fnn_activation_nonlinear

    implicit none

    private
    public :: ReluActivation, construct_relu_activation

    !--------------------------------------------------
    !> @brief Implements a `relu` activation function.
    type, extends(NonLinearActivation) :: ReluActivation
    contains
        !> @brief Saves the activation function.
        !> @details Implemented by \ref relu_tofile.
        procedure, pass, public :: tofile => relu_tofile
        !> @brief Applies and linearises the activation function.
        !> @details Implemented by \ref relu_apply_forward.
        procedure, pass, public :: apply_forward => relu_apply_forward
    end type ReluActivation

contains

    !--------------------------------------------------
    !> @brief Constructor for class \ref reluactivation.
    !> @param[in] self_size The value for \ref linearactivation::self_size.
    !> @param[in] batch_size The value for \ref linearactivation::batch_size.
    !> @return The constructed activation function.
    type(ReluActivation) function construct_relu_activation(self_size, batch_size) result(self)
        integer(ik), intent(in) :: self_size
        integer(ik), intent(in) :: batch_size
        self % NonLinearActivation = construct_nonlinear_activation(self_size, batch_size)
    end function construct_relu_activation

    !--------------------------------------------------
    !> @brief Implements \ref reluactivation::tofile.
    !>
    !> Saves the activation function.
    !> @param[in] self The activation function to save.
    !> @param[in] unit_num The unit number for the write statement.
    subroutine relu_tofile(self, unit_num)
        class(ReluActivation), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        write(unit_num, fmt=*) 'relu'
    end subroutine relu_tofile

    !--------------------------------------------------
    !> @brief Implements \ref reluactivation::apply_forward.
    !>
    !> Applies and linearises the activation function.
    !> @details The activation function reads
    !> \f[ \mathbf{y} = \mathcal{A}(\mathbf{z}) = \mathrm{relu}(\mathbf{z}),\f]
    !> and the associated linearisation reads
    !> \f[ \mathbf{A}(\mathbf{z}) = \mathrm{diag}(1-\mathrm{relu}(\mathbf{z})^2).\f]
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
    subroutine relu_apply_forward(self, member, z, y)
        class(ReluActivation), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: z(:)
        real(rk), intent(out) :: y(:)
        integer(ik) :: i
        do i = 1, size(y)
            if (z(i) > 0) then
                y(i) = z(i)
                self % z_prime(i, member) = 1
            else
                y(i) = 0
                self % z_prime(i, member) = 0
            end if
        end do
    end subroutine relu_apply_forward

end module fnn_activation_relu

