
!> @brief Module dedicated to the class \ref dropoutlayer.
module fnn_layer_dropout

    use fnn_common
    use fnn_activation_linear
    use fnn_activation_tanh
    use fnn_activation_relu
    use fnn_layer

    implicit none

    private
    public :: DropoutLayer, dropout_layer_fromfile

    !--------------------------------------------------
    !> @brief Implements a dropout layer.
    !> @details This layer has no (trainable) parameters.
    type, extends(Layer) :: DropoutLayer
        private
        !> The dropout probability.
        real(rk) :: rate
        !> @brief The storage for the linearisation.
        !> @details Array of reals with shape (layer::input_size, layer::batch_size).
        real(rk), allocatable :: draws(:, :)
    contains
        !> @brief Saves the layer.
        !> Implemented by \ref dropout_tofile.
        procedure, pass, public :: tofile => dropout_tofile
        !> @brief Applies and linearises the layer.
        !> Implemented by \ref dropout_apply_forward.
        procedure, pass, public :: apply_forward => dropout_apply_forward
        !> @brief Applies the TL of the layer.
        !> Implemented by \ref dropout_apply_tangent_linear.
        procedure, pass, public :: apply_tangent_linear => dropout_apply_tangent_linear
        !> @brief Applies the adjoint of the layer.
        !> Implemented by \ref dropout_apply_adjoint.
        procedure, pass, public :: apply_adjoint => dropout_apply_adjoint
    end type DropoutLayer

contains

    !--------------------------------------------------
    !> @brief Constructor for class \ref dropoutlayer from a file.
    !> @param[in] batch_size The value for layer::batch_size.
    !> @param[in] unit_num The unit number for the read statements.
    !> @return The constructed layer.
    type(DropoutLayer) function dropout_layer_fromfile(batch_size, unit_num) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: unit_num
        read(unit_num, *) self % input_size
        read(unit_num, *) self % rate
        self % output_size = self % input_size
        self % batch_size = batch_size
        self % num_parameters = 0
        allocate(self % parameters(0))
        allocate(self % forward_input(self % input_size, self % batch_size))
        allocate(self % tangent_linear_input(self % input_size, self % batch_size))
        allocate(self % adjoint_input(self % output_size, self % batch_size))
        allocate(self % draws(self % input_size, self % batch_size))
        self % forward_input = 0
        self % tangent_linear_input = 0
        self % adjoint_input = 0
        self % draws = 0
    end function dropout_layer_fromfile

    !--------------------------------------------------
    !> @brief Implements \ref dropoutlayer::tofile.
    !>
    !> Saves the layer.
    !> @param[in] self The layer.
    !> @param[in] unit_num The unit number for the write statement.
    subroutine dropout_tofile(self, unit_num)
        class(DropoutLayer), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        write(unit_num, fmt=*) 'dropout'
        write(unit_num, fmt=*) self % input_size
        write(unit_num, fmt=*) self % rate
    end subroutine dropout_tofile

    !--------------------------------------------------
    !> @brief Implements \ref dropoutlayer::apply_forward.
    !>
    !> Applies and linearises the layer.
    !>
    !> @details The forward function reads
    !> \f[\mathbf{y} = \mathbf{z}*\mathbf{x},\f]
    !> where \f$*\f$ is the element-wise multiplication 
    !> for vectors.
    !> 
    !> In training mode, \f$z_{i}=0\f$ with probability \f$p\f$
    !> and \f$z_{i}=1/(1-p)\f$ with probability \f$1-p\f$ where
    !> \f$p\f$ is the dropout rate.
    !>
    !> In testing mode, \f$z_{i}=1\f$.
    !>
    !> \b Note
    !>
    !> Input parameter `member` should be less than layer::batch_size.
    !>
    !> The linearisation is trivial and only requires to store the values
    !> of \f$\mathbf{z}\f$ in \ref dropoutlayer::draws.
    !> @param[inout] self The layer.
    !> @param[in] train Whether the model is used in training mode.
    !> @param[in] member The index inside the batch.
    !> @param[in] x The input of the layer.
    !> @param[out] y The output of the layer.
    subroutine dropout_apply_forward(self, train, member, x, y)
        class(DropoutLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        integer(ik) :: n
        if ( train ) then
            call rand1d(self % draws(:, member))
            do n = 1, self % input_size
                if ( self % draws(n, member) < self % rate ) then 
                    self % draws(n, member) = 0
                else
                    self % draws(n, member) = 1 / (1 - self%rate)
                endif
            enddo
        else
            self % draws(:, member) = 1
        endif
        y = self % draws(:, member) * x
    end subroutine dropout_apply_forward

    !--------------------------------------------------
    !> @brief Implements \ref dropoutlayer::apply_tangent_linear.
    !>
    !> Applies the TL of the layer.
    !>
    !> @details  The TL operator reads
    !> \f[d\mathbf{y} = \mathbf{z}*d\mathbf{x}.\f]
    !>
    !> \b Note
    !>
    !> This method should only be called 
    !> after \ref dropoutlayer::apply_forward.
    !>
    !> Since there is no (trainable) parameters, the
    !> parameter perturbation should be an empty array.
    !> @param[in] self The layer.
    !> @param[in] member The index inside the batch.
    !> @param[in] dp The parameter perturbation.
    !> @param[in] dx The state perturbation.
    !> @param[out] dy The output perturbation.
    subroutine dropout_apply_tangent_linear(self, member, dp, dx, dy)
        class(DropoutLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = self % draws(:, member) * dx
    end subroutine dropout_apply_tangent_linear

    !--------------------------------------------------
    !> @brief Implements \ref dropoutlayer::apply_adjoint.
    !>
    !> Applies the adjoint of the layer.
    !>
    !> @details  The adjoint operator reads
    !> \f[d\mathbf{x} = \mathbf{z}*d\mathbf{y}.\f]
    !>
    !> \b Note
    !>
    !> This method should only be called after
    !> \ref dropoutlayer::apply_forward.
    !>
    !> Since there is no (trainable) parameters, the
    !> parameter perturbation should be an empty array.
    !>
    !> The intent of `dy` is declared `inout` instead of `in` because of other
    !> subclasses of \ref fnn_layer::layer.
    !> @param[inout] self The layer.
    !> @param[in] member The index inside the batch.
    !> @param[inout] dy The output perturbation.
    !> @param[out] dp The parameter perturbation.
    !> @param[out] dx The state perturbation.
    subroutine dropout_apply_adjoint(self, member, dy, dp, dx)
        class(DropoutLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        dx = self % draws(:, member) * dy
    end subroutine dropout_apply_adjoint

end module fnn_layer_dropout

