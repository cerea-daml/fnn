
!> @brief Module dedicated to the class \ref frozendenselayer.
module fnn_layer_frozen_dense

    use fnn_common
    use fnn_activation_linear
    use fnn_activation_tanh
    use fnn_activation_relu
    use fnn_layer

    implicit none

    private
    public :: FrozenDenseLayer, construct_frozen_dense_layer, frozen_dense_layer_fromfile

    !--------------------------------------------------
    !> @brief Implements a frozen dense (fully-connected) layer.
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
    !> \ref frozen_dense_apply_tangent_linear and \ref frozen_dense_apply_adjoint
    !> methods.
    type, extends(Layer) :: FrozenDenseLayer
        private
        !> The frozen parameters.
        real(rk), allocatable :: frozen_parameters(:)
    contains
        !> @brief Reads the parameters from binary file.
        !> Implemented by \ref frozen_dense_read_parameters.
        procedure, pass, public :: read_parameters => frozen_dense_read_parameters
        !> @brief Saves the layer.
        !> Implemented by \ref frozen_dense_tofile.
        procedure, pass, public :: tofile => frozen_dense_tofile
        !> @brief Applies and linearises the layer.
        !> Implemented by \ref frozen_dense_apply_forward.
        procedure, pass, public :: apply_forward => frozen_dense_apply_forward
        !> @brief Applies the TL of the layer.
        !> Implemented by \ref frozen_dense_apply_tangent_linear.
        procedure, pass, public :: apply_tangent_linear => frozen_dense_apply_tangent_linear
        !> @brief Applies the adjoint of the layer.
        !> Implemented by \ref frozen_dense_apply_adjoint.
        procedure, pass, public :: apply_adjoint => frozen_dense_apply_adjoint
    end type FrozenDenseLayer

contains

    !--------------------------------------------------
    !> @brief Manual constructor for class \ref frozendenselayer.
    !> Only for testing purpose.
    !> @param[in] input_size The value for layer::input_size.
    !> @param[in] output_size The value for layer::output_size.
    !> @param[in] batch_size The value for layer::batch_size.
    !> @param[in] activation_name The activation function.
    !> @param[in] initialisation_name The initialisation for model parameters.
    !> @return The constructed layer.
    type(FrozenDenseLayer) function construct_frozen_dense_layer(input_size, output_size,&
            batch_size, activation_name, initialisation_name) result(self)
        integer(ik), intent(in) :: input_size
        integer(ik), intent(in) :: output_size
        integer(ik), intent(in) :: batch_size
        character(len=*), intent(in) :: activation_name
        character(len=*), intent(in) :: initialisation_name
        self % input_size = input_size
        self % output_size = output_size
        self % batch_size = batch_size
        self % num_parameters = 0
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
        allocate(self % parameters(0))
        allocate(self % frozen_parameters((input_size+1) * output_size))
        allocate(self % forward_input(input_size, batch_size))
        allocate(self % tangent_linear_input(input_size, batch_size))
        allocate(self % adjoint_input(output_size, batch_size))
        select case(trim(initialisation_name))
            case('rand')
                call rand1d(self % frozen_parameters)
            case default
                self % frozen_parameters = 0
        end select
        self % forward_input = 0
        self % tangent_linear_input = 0
        self % adjoint_input = 0
    end function construct_frozen_dense_layer

    !--------------------------------------------------
    !> @brief Constructor for class \ref frozendenselayer from a file.
    !> @param[in] batch_size The value for layer::batch_size.
    !> @param[in] unit_num The unit number for the read statements.
    !> @return The constructed layer.
    type(FrozenDenseLayer) function frozen_dense_layer_fromfile(batch_size, unit_num) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: unit_num
        character(len=100) :: activation_name
        read(unit_num, *) self % input_size
        read(unit_num, *) self % output_size
        self % batch_size = batch_size
        self % num_parameters = 0
        allocate(self % frozen_parameters((self % input_size+1) * self % output_size))
        allocate(self % parameters(0))
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
        self % frozen_parameters = 0
        self % forward_input = 0
        self % tangent_linear_input = 0
        self % adjoint_input = 0
    end function frozen_dense_layer_fromfile

    !--------------------------------------------------
    !> @brief Implements \ref frozendenselayer::read_parameters.
    !>
    !> Reads the parameters from binary file.
    !> @param[inout] self The layer.
    !> @param[in] unit_num The unit number for the read statement.
    subroutine frozen_dense_read_parameters(self, unit_num)
        class(FrozenDenseLayer), intent(inout) :: self
        integer(ik), intent(in) :: unit_num
        real(r0), allocatable :: the_parameters(:)
        ! read in r0 precision
        allocate(the_parameters(size(self % frozen_parameters)))
        read(unit_num) the_parameters
        ! cast to rk precision
        self % frozen_parameters = the_parameters
    end subroutine frozen_dense_read_parameters

    !--------------------------------------------------
    !> @brief Implements \ref frozendenselayer::tofile.
    !>
    !> Saves the layer. (Not the parameters)
    !> @param[in] self The layer.
    !> @param[in] unit_num The unit number for the write statement.
    subroutine frozen_dense_tofile(self, unit_num)
        class(FrozenDenseLayer), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        write(unit_num, fmt=*) 'frozen-dense'
        write(unit_num, fmt=*) self % input_size
        write(unit_num, fmt=*) self % output_size
        call self % activation % tofile(unit_num)
    end subroutine frozen_dense_tofile

    !--------------------------------------------------
    !> @brief Implements \ref frozendenselayer::apply_forward.
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
    !> @param[in] train Whether the model is used in training mode.
    !> @param[in] member The index inside the batch.
    !> @param[in] x The input of the layer.
    !> @param[out] y The output of the layer.
    subroutine frozen_dense_apply_forward(self, train, member, x, y)
        class(FrozenDenseLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        self % forward_input(:, member) = x
        y = matmul(&
            reshape(self % frozen_parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            x)
        y = y + self % frozen_parameters(1:self % output_size)
        call self % activation % apply_forward(member, y, y)
    end subroutine frozen_dense_apply_forward

    !--------------------------------------------------
    !> @brief Implements \ref frozendenselayer::apply_tangent_linear.
    !>
    !> Applies the TL of the layer.
    !>
    !> @details  The TL operator reads
    !> \f[d\mathbf{y} = \mathbf{A}(\mathbf{Wx+b})[\mathbf{W}d
    !> \mathbf{x}],\f]
    !> which is implemented by
    !> \f[d\mathbf{y} = \mathbf{W}d\mathbf{x},\f]
    !> \f[d\mathbf{y} = \mathbf{A}(\mathbf{Wx+b})d\mathbf{y}.\f]
    !>
    !> \b Note
    !>
    !> This method should only be called after
    !> \ref frozendenselayer::apply_forward.
    !> @param[in] self The layer.
    !> @param[in] member The index inside the batch.
    !> @param[in] dp The parameter perturbation.
    !> @param[in] dx The state perturbation.
    !> @param[out] dy The output perturbation.
    subroutine frozen_dense_apply_tangent_linear(self, member, dp, dx, dy)
        class(FrozenDenseLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = matmul(&
            reshape(self % frozen_parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            dx)
        call self % activation % apply_tangent_linear(member, dy, dy)
    end subroutine frozen_dense_apply_tangent_linear

    !--------------------------------------------------
    !> @brief Implements \ref frozendenselayer::apply_adjoint.
    !>
    !> Applies the adjoint of the layer.
    !>
    !> @details The adjoint operator is implemented by
    !> \f[d\mathbf{y} = \mathbf{A}(\mathbf{Wx+b})^{\top}d\mathbf{y},\f]
    !> \f[d\mathbf{x} = \mathbf{W}^{\top}d\mathbf{y}.\f]
    !>
    !> \b Note
    !>
    !> This method should only be called after
    !> \ref frozendenselayer::apply_forward.
    !> 
    !> The value of \f$d\mathbf{y}\f$ gets overwritten in this method
    !> (bad side-effect). 
    !> For this reason, the intent of `dy` is declared `inout`.
    !> @param[inout] self The layer.
    !> @param[in] member The index inside the batch.
    !> @param[inout] dy The output perturbation.
    !> @param[out] dp The parameter perturbation.
    !> @param[out] dx The state perturbation.
    subroutine frozen_dense_apply_adjoint(self, member, dy, dp, dx)
        class(FrozenDenseLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        call self % activation % apply_adjoint(member, dy, dy)
        dx = matmul(transpose(reshape(self % frozen_parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size])), dy)
    end subroutine frozen_dense_apply_adjoint

end module fnn_layer_frozen_dense

