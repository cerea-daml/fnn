
!> @brief Module dedicated to the class \ref denselayer.
module fnn_layer_dense

    use fnn_common
    use fnn_activation_linear
    use fnn_activation_tanh
    use fnn_activation_relu
    use fnn_layer

    implicit none

    private
    public :: DenseLayer, construct_dense_layer, dense_layer_fromfile

    !--------------------------------------------------
    !> @brief Implements a dense (fully-connected) layer.
    !> @details This layer has two sets of (trainable) parameters:
    !> - the kernel \f$\mathbf{W}\f$, a matrix of size (layer::output_size, layer::input_size);
    !> - the bias \f$\mathbf{b}\f$, a vector of size (layer::output_size).
    !>
    !> Using numpy syntax, \f$\mathbf{b}\f$ and \f$\mathbf{W}\f$ are obtained
    !> from \f$\mathbf{p}\f$ through
    !> \f[ \mathbf{b} = \mathbf{p}[:N_{\mathrm{out}}],\f]
    !> \f[ \mathbf{W} = \mathbf{p}[N_{\mathrm{out}}:].\mathrm{reshape}((N_{\mathrm{out}}, 
    !> N_{\mathrm{in}}), \mathrm{order="F"}),\f]
    !> where \f$N_{\mathrm{out}}\f$ is layer::output_size and
    !> \f$N_{\mathrm{in}}\f$ is layer::input_size.
    !>
    !> A similar relationship holds between \f$d\mathbf{b}\f$,
    !> \f$d\mathbf{W}\f$ and \f$d\mathbf{p}\f$ in the
    !> \ref dense_apply_tangent_linear and \ref dense_apply_adjoint
    !> methods.
    type, extends(Layer) :: DenseLayer
        private
    contains
        !> @brief Saves the layer.
        !> Implemented by \ref dense_tofile.
        procedure, pass, public :: tofile => dense_tofile
        !> @brief Applies and linearises the layer.
        !> Implemented by \ref dense_apply_forward.
        procedure, pass, public :: apply_forward => dense_apply_forward
        !> @brief Applies the TL of the layer.
        !> Implemented by \ref dense_apply_tangent_linear.
        procedure, pass, public :: apply_tangent_linear => dense_apply_tangent_linear
        !> @brief Applies the adjoint of the layer.
        !> Implemented by \ref dense_apply_adjoint.
        procedure, pass, public :: apply_adjoint => dense_apply_adjoint
    end type DenseLayer

contains

    !--------------------------------------------------
    !> @brief Manual constructor for class \ref denselayer.
    !> Only for testing purpose.
    !> @param[in] input_size The value for layer::input_size.
    !> @param[in] output_size The value for layer::output_size.
    !> @param[in] batch_size The value for layer::batch_size.
    !> @param[in] activation_name The activation function.
    !> @param[in] initialisation_name The initialisation for model parameters.
    !> @return The constructed layer.
    type(DenseLayer) function construct_dense_layer(input_size, output_size,&
            batch_size, activation_name, initialisation_name) result(self)
        integer(ik), intent(in) :: input_size
        integer(ik), intent(in) :: output_size
        integer(ik), intent(in) :: batch_size
        character(len=*), intent(in) :: activation_name
        character(len=*), intent(in) :: initialisation_name
        self % input_size = input_size
        self % output_size = output_size
        self % batch_size = batch_size
        self % num_parameters = (input_size+1) * output_size
        select case(trim(activation_name))
            case('tanh')
                allocate(TanhActivation::self % activation)
                self % activation = construct_tanh_activation(output_size, batch_size)
            case('relu')
                allocate(ReluActivation::self % activation)
                self % activation = construct_relu_activation(output_size, batch_size)
            case default
                allocate(LinearActivation::self % activation)
                self % activation = construct_linear_activation(output_size, batch_size)
        end select
        allocate(self % parameters(self % num_parameters))
        allocate(self % forward_input(input_size, batch_size))
        allocate(self % tangent_linear_input(input_size, batch_size))
        allocate(self % adjoint_input(output_size, batch_size))
        select case(trim(initialisation_name))
            case('rand')
                call rand1d(self % parameters)
            case default
                self % parameters = 0
        end select
        self % forward_input = 0
        self % tangent_linear_input = 0
        self % adjoint_input = 0
    end function construct_dense_layer

    !--------------------------------------------------
    !> @brief Constructor for class \ref denselayer from a file.
    !> @param[in] batch_size The value for layer::batch_size.
    !> @param[in] unit_num The unit number for the read statements.
    !> @return The constructed layer.
    type(DenseLayer) function dense_layer_fromfile(batch_size, unit_num) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: unit_num
        character(len=100) :: activation_name
        read(unit_num, *) self % input_size
        read(unit_num, *) self % output_size
        self % batch_size = batch_size
        self % num_parameters = (self % input_size+1) * self % output_size
        allocate(self % parameters(self % num_parameters))
        read(unit_num, *) self % parameters
        read(unit_num, *) activation_name
        select case(trim(activation_name))
            case('tanh')
                allocate(TanhActivation::self % activation)
                self % activation = construct_tanh_activation(self % output_size, self % batch_size)
            case('relu')
                allocate(ReluActivation::self % activation)
                self % activation = construct_relu_activation(self % output_size, self % batch_size)
            case default
                allocate(LinearActivation::self % activation)
                self % activation = construct_linear_activation(self % output_size, self % batch_size)
        end select
        allocate(self % forward_input(self % input_size, self % batch_size))
        allocate(self % tangent_linear_input(self % input_size, self % batch_size))
        allocate(self % adjoint_input(self % output_size, self % batch_size))
        self % forward_input = 0
        self % tangent_linear_input = 0
        self % adjoint_input = 0
    end function dense_layer_fromfile

    !--------------------------------------------------
    !> @brief Implements \ref denselayer::tofile.
    !>
    !> Saves the layer.
    !> @param[in] self The layer.
    !> @param[in] unit_num The unit number for the write statement.
    subroutine dense_tofile(self, unit_num)
        class(DenseLayer), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        write(unit_num, fmt=*) 'dense'
        write(unit_num, fmt=*) self % input_size
        write(unit_num, fmt=*) self % output_size
        write(unit_num, fmt=*) self % parameters
        call self % activation % tofile(unit_num)
    end subroutine dense_tofile

    !--------------------------------------------------
    !> @brief Implements \ref denselayer::apply_forward.
    !>
    !> Applies and linearises the layer.
    !>
    !> @details The forward function reads
    !> \f[ \mathbf{y} = \mathcal{F}(\mathbf{p}, \mathbf{x})
    !> = \mathcal{A}(\mathbf{Wx+b}),\f]
    !> where \f$\mathbf{W}\f$ is the kernel and \f$\mathbf{b}\f$
    !> the bias of the layer, and where \f$\mathcal{A}\f$ is
    !> the activation function.
    !>
    !> \b Note
    !>
    !> Input parameter `member` should be less than layer::batch_size.
    !>
    !> The linearisation of the regression \f$\mathbf{Wx+b}\f$ is 
    !> stored in layer::forward_input, and the linearisation of
    !> the activation function is stored in layer::activation.
    !>
    !> Because the linearisation of the layer is stored inside
    !> the layer, the intent of `self`is declared `inout`.
    !> @todo Find a way to store (internally) a view to the
    !> kernel and the bias.
    !> @param[inout] self The layer.
    !> @param[in] member The index inside the batch.
    !> @param[in] x The input of the layer.
    !> @param[out] y The output of the layer.
    subroutine dense_apply_forward(self, member, x, y)
        class(DenseLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        self % forward_input(:, member) = x
        y = matmul(&
            reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            x)
        y = y + self % parameters(1:self % output_size)
        call self % activation % apply_forward(member, y, y)
    end subroutine dense_apply_forward

    !--------------------------------------------------
    !> @brief Implements \ref denselayer::apply_tangent_linear.
    !>
    !> Applies the TL of the layer.
    !>
    !> @details  The TL operator reads
    !> \f[d\mathbf{y} = \mathbf{A}(\mathbf{Wx+b})[\mathbf{W}d
    !> \mathbf{x}+d\mathbf{Wx}+d\mathbf{b}],\f]
    !> which is implemented by
    !> \f[d\mathbf{y} = \mathbf{W}d\mathbf{x},\f]
    !> \f[d\mathbf{y} = d\mathbf{y} + d\mathbf{Wx},\f]
    !> \f[d\mathbf{y} = d\mathbf{y} + d\mathbf{b},\f]
    !> \f[d\mathbf{y} = \mathbf{A}(\mathbf{Wx+b})d\mathbf{y}.\f]
    !>
    !> \b Note
    !>
    !> This method should only be called after
    !> \ref denselayer::apply_forward.
    !> @param[in] self The layer.
    !> @param[in] member The index inside the batch.
    !> @param[in] dp The parameter perturbation.
    !> @param[in] dx The state perturbation.
    !> @param[out] dy The output perturbation.
    subroutine dense_apply_tangent_linear(self, member, dp, dx, dy)
        class(DenseLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = matmul(&
            reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            dx)
        dy = dy + matmul(&
            reshape(dp(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            self % forward_input(:, member))
        dy = dy + dp(1:self % output_size)
        call self % activation % apply_tangent_linear(member, dy, dy)
    end subroutine dense_apply_tangent_linear

    !--------------------------------------------------
    !> @brief Implements \ref denselayer::apply_adjoint.
    !>
    !> Applies the adjoint of the layer.
    !>
    !> @details The adjoint operator is implemented by
    !> \f[d\mathbf{y} = \mathbf{A}(\mathbf{Wx+b})^{\top}d\mathbf{y},\f]
    !> \f[d\mathbf{b} = d\mathbf{y},\f]
    !> \f[d\mathbf{W} = d\mathbf{yx}^{\top},\f]
    !> \f[d\mathbf{x} = \mathbf{W}^{\top}d\mathbf{y}.\f]
    !>
    !> \b Note
    !>
    !> This method should only be called after
    !> \ref denselayer::apply_forward.
    !> 
    !> The value of \f$d\mathbf{y}\f$ gets overwritten in this method
    !> (bad side-effect). This could be easily solved by merging the first
    !> two algorithmic lines into
    !> \f[d\mathbf{b} = \mathbf{A}(\mathbf{Wx+b})^{\top}d\mathbf{y}.\f]
    !> For this reason, the intent of `dy` is declared `inout`.
    !> @param[inout] self The layer.
    !> @param[in] member The index inside the batch.
    !> @param[inout] dy The output perturbation.
    !> @param[out] dp The parameter perturbation.
    !> @param[out] dx The state perturbation.
    subroutine dense_apply_adjoint(self, member, dy, dp, dx)
        class(DenseLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        call self % activation % apply_adjoint(member, dy, dy)
        dp(1:self % output_size) = dy
        dp(self % output_size+1:self % output_size*(self % input_size+1)) = reshape(matmul(&
            reshape(dp(1:self % output_size), [self % output_size, 1]),&
            reshape(self % forward_input(:, member), [1, self % input_size])),&
            [self % output_size*self % input_size])
        dx = matmul(transpose(reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size])), dp(1:self % output_size))
    end subroutine dense_apply_adjoint

end module fnn_layer_dense

