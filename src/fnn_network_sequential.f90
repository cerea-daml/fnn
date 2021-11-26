
!> @brief Module dedicated to the class \ref sequentialneuralnetwork.
module fnn_network_sequential

    use fnn_common
    use fnn_layer
    use fnn_layer_dense
    use fnn_layer_normalisation

    implicit none

    private
    public :: SequentialNeuralNetwork, construct_sequential_neural_network, snn_fromfile

    !--------------------------------------------------
    !> @brief Layer container class.
    !> @details This class only exists to enable polymorphism.
    type :: LayerContainer
        !> The contained layer.
        class(Layer), allocatable :: this_layer
    end type

    !--------------------------------------------------
    !> @brief Implements a sequential neural network.
    type :: SequentialNeuralNetwork
        private
        !> The batch size.
        integer(ik) :: batch_size
        !> The number of layers.
        integer(ik) :: num_layers
        !> The total number of parameters.
        integer(ik) :: num_parameters
        !> The list of layers.
        class(LayerContainer), allocatable :: list_layers(:)
        !> @brief The starting indices for the parameter repartition.
        !> @details If \f$\mathbf{p}\f$ is an array of size sequentialneuralnetwork::num_parameters
        !> containing the parameters for all layers, 
        !> then the slice of \f$\mathbf{p}\f$ starting at `ip_start(i)` and
        !> ending at `ip_end(i)` contains the parameters
        !> for the `i`-th layer.
        integer(ik), allocatable :: ip_start(:)
        !> @brief The ending indices for the parameter repartition.
        !> @details See sequentialneuralnetwork::ip_start.
        integer(ik), allocatable :: ip_end(:)
    contains
        !> @brief Returns the input size of the network.
        !> Implemented by \ref snn_get_input_size.
        procedure, public, pass :: get_input_size => snn_get_input_size
        !> @brief Returns the output size of the network.
        !> Implemented by \ref snn_get_output_size.
        procedure, public, pass :: get_output_size => snn_get_output_size
        !> @brief Returns the number of parameters.
        !> Implemented by \ref snn_get_num_parameters.
        procedure, public, pass :: get_num_parameters => snn_get_num_parameters
        !> @brief Setter for the network's parameters.
        !> Implemented by \ref snn_set_parameters.
        procedure, public, pass :: set_parameters => snn_set_parameters
        !> @brief Getter for the network's parameters.
        !> Implemented by \ref snn_get_parameters.
        procedure, public, pass :: get_parameters => snn_get_parameters
        !> @brief Saves the network.
        !> Implemented by \ref snn_tofile.
        procedure, public, pass :: tofile => snn_tofile
        !> @brief Applies and linearises the network.
        !> Implemented by \ref snn_apply_forward.
        procedure, public, pass :: apply_forward => snn_apply_forward
        !> @brief Applies the TL of the network.
        !> Implemented by \ref snn_apply_tangent_linear.
        procedure, public, pass :: apply_tangent_linear => snn_apply_tangent_linear
        !> @brief Applies the adjoint of the networ.
        !> Implemented by \ref snn_apply_adjoint.
        procedure, public, pass :: apply_adjoint => snn_apply_adjoint
    end type SequentialNeuralNetwork

contains

    !--------------------------------------------------
    !> @brief Manual constructor for class \ref sequentialneuralnetwork.
    !> Only for testing purpose.
    !> @param[in] batch_size The value for sequentialneuralnetwork::batch_size.
    !> @param[in] internal_sizes The list of internal sizes.
    !> @param[in] activation_names The list of activation functions.
    !> @param[in] initialisation_names The list of initialisations for model parameters.
    !> @return The constructed network.
    type(SequentialNeuralNetwork) function& 
            construct_sequential_neural_network(batch_size, internal_sizes, activation_names,&
            initialisation_names) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: internal_sizes(:)
        character(len=*), intent(in) :: activation_names(:)
        character(len=*), intent(in) :: initialisation_names(:)
        integer(ik) :: i
        integer(ik) :: ip
        self % batch_size = batch_size
        self % num_layers = size(internal_sizes) - 1
        allocate(self % list_layers(self % num_layers))
        allocate(self % ip_start(self % num_layers))
        allocate(self % ip_end(self % num_layers))
        ip = 0
        do i = 1, self % num_layers
            allocate(DenseLayer::self % list_layers(i) % this_layer)
            self % list_layers(i) % this_layer = construct_dense_layer(internal_sizes(i), internal_sizes(i+1),&
                batch_size, activation_names(i), initialisation_names(i))
            self % ip_start(i) = ip + 1
            ip = ip + self % list_layers(i) % this_layer % get_num_parameters()
            self % ip_end(i) = ip
        end do
        self % num_parameters = ip
    end function construct_sequential_neural_network

    !--------------------------------------------------
    !> @brief Constructor for class \ref sequentialneuralnetwork from a file.
    !> @param[in] batch_size The value for layer::batch_size.
    !> @param[in] filename The name of the file to read.
    !> @return The constructed network.
    type(SequentialNeuralNetwork) function snn_fromfile(batch_size, filename) result(self)
        integer(ik), intent(in) :: batch_size
        character(len=*), intent(in) :: filename
        integer(ik) :: fileunit
        character(len=100) :: network_name
        character(len=100) :: layer_name
        integer(ik) :: i
        integer(ik) :: ip
        open(newunit=fileunit, file=filename, action='read')
        read(fileunit, fmt=*) network_name
        if ( trim(network_name) == 'sequential' ) then
            read(fileunit, fmt=*) self % num_layers
            allocate(self % list_layers(self % num_layers))
            allocate(self % ip_start(self % num_layers))
            allocate(self % ip_end(self % num_layers))
            ip = 0
            do i = 1, self % num_layers
                read(fileunit, fmt=*) layer_name
                select case(trim(layer_name))
                    case('normalisation')
                        allocate(NormalisationLayer::self % list_layers(i) % this_layer)
                        self % list_layers(i) % this_layer = norm_layer_fromfile(batch_size, fileunit)
                        self % ip_start(i) = ip + 1
                        self % ip_end(i) = ip
                    case default ! default to dense layer
                        allocate(DenseLayer::self % list_layers(i) % this_layer)
                        self % list_layers(i) % this_layer = dense_layer_fromfile(batch_size, fileunit)
                        self % ip_start(i) = ip + 1
                        ip = ip + self % list_layers(i) % this_layer % get_num_parameters()
                        self % ip_end(i) = ip
                end select
            end do
            self % num_parameters = ip
        else
            print *, 'ERROR: unknown network name (', trim(network_name), ')'
        end if
        close(fileunit)
    end function snn_fromfile

    !--------------------------------------------------
    !> @brief Implements \ref sequentialneuralnetwork::get_input_size.
    !>
    !> Returns the input size of the network.
    !>
    !> @param[in] self The network.
    !> @return The input size.
    integer(ik) function snn_get_input_size(self) result(input_size)
        class(SequentialNeuralNetwork), intent(in) :: self
        input_size = self % list_layers(1) % this_layer % input_size
    end function snn_get_input_size

    !--------------------------------------------------
    !> @brief Implements \ref sequentialneuralnetwork::get_output_size.
    !>
    !> Returns the output size of the network.
    !>
    !> @param[in] self The network.
    !> @return The output size.
    integer(ik) function snn_get_output_size(self) result(output_size)
        class(SequentialNeuralNetwork), intent(in) :: self
        output_size = self % list_layers(self % num_layers) % this_layer % output_size
    end function snn_get_output_size

    !--------------------------------------------------
    !> @brief Implements \ref sequentialneuralnetwork::get_num_parameters.
    !>
    !> Returns the number of parameters.
    !>
    !> @param[in] self The network.
    !> @return The number of parameters.
    integer(ik) function snn_get_num_parameters(self) result(num_parameters)
        class(SequentialNeuralNetwork), intent(in) :: self
        num_parameters = self % num_parameters
    end function snn_get_num_parameters

    !--------------------------------------------------
    !> @brief Implements \ref sequentialneuralnetwork::set_parameters.
    !>
    !> Setter for the network's parameters.
    !>
    !> @param[inout] self The network.
    !> @param[in] new_parameters The new values for the parameters.
    subroutine snn_set_parameters(self, new_parameters)
        class(SequentialNeuralNetwork), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        integer(ik) :: i
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % set_parameters(&
                new_parameters(self % ip_start(i):self % ip_end(i)))
        end do
    end subroutine snn_set_parameters

    !--------------------------------------------------
    !> @brief Implements \ref sequentialneuralnetwork::get_parameters.
    !>
    !> Getter for the network's parameters.
    !>
    !> @param[in] self The network.
    !> @param[out] parameters The vector of parameters.
    subroutine snn_get_parameters(self, parameters)
        class(SequentialNeuralNetwork), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        integer(ik) :: i
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % get_parameters(&
                parameters(self % ip_start(i):self % ip_end(i)))
        end do
    end subroutine snn_get_parameters

    !--------------------------------------------------
    !> @brief Implements \ref sequentialneuralnetwork::tofile.
    !>
    !> Saves the network.
    !> @param[in] self The network.
    !> @param[in] filename The name of the file to write.
    subroutine snn_tofile(self, filename)
        class(SequentialNeuralNetwork), intent(in) :: self
        character(len=*), intent(in) :: filename
        integer(ik) :: fileunit
        integer(ik) :: i
        open(newunit=fileunit, file=filename, action='write')
        write(fileunit, fmt=*) 'sequential'
        write(fileunit, fmt=*) self % num_layers
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % tofile(fileunit)
        end do
        close(fileunit)
    end subroutine snn_tofile

    !--------------------------------------------------
    !> @brief Implements \ref sequentialneuralnetwork::apply_forward.
    !>
    !> Applies and linearises the network.
    !>
    !> @details \b Note
    !>
    !> If there is more than one layer, this method uses
    !> layer::forward_input as intermediate storage.
    !>
    !> For the case of a \ref fnn_layer_dense::denselayer, this means
    !> that the first step of denselayer::apply_forward
    !> consists in copying layer::forward_input into...
    !> layer::forward_input!
    !>
    !> For theses reasons, and because the linearisation is
    !> stored inside the network, the intent of `self` is
    !> declared `inout`.
    !> @param[inout] self The network.
    !> @param[in] member The index inside the batch.
    !> @param[in] x The input of the network.
    !> @param[out] y The output of the network.
    subroutine snn_apply_forward(self, member, x, y)
        class(SequentialNeuralNetwork), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        integer(ik) :: i
        if ( self % num_layers == 1 ) then
            call self % list_layers(1) % this_layer % apply_forward(member, x, y)
        else
            call self % list_layers(1) % this_layer % apply_forward(member, x,&
                self % list_layers(2) % this_layer % forward_input(:, member))
            do i = 2, self % num_layers - 1
                call self % list_layers(i) % this_layer % apply_forward(member,&
                    self % list_layers(i) % this_layer % forward_input(:, member),&
                    self % list_layers(i+1) % this_layer % forward_input(:, member))
            end do
            call self % list_layers(self % num_layers) % this_layer % apply_forward(member,&
                self % list_layers(self % num_layers) % this_layer % forward_input(:, member), y)
        end if
    end subroutine snn_apply_forward

    !--------------------------------------------------
    !> @brief Implements \ref sequentialneuralnetwork::apply_tangent_linear.
    !>
    !> Applies the TL of the network.
    !>
    !> @details \b Note
    !>
    !> If there is more than one layer, this method uses
    !> layer::tangent_linear_input as intermediate storage.
    !>
    !> For this reason, the intent of `self` is declared `inout`.
    !> @param[inout] self The network.
    !> @param[in] member The index inside the batch.
    !> @param[in] dp The parameter perturbation.
    !> @param[in] dx The state perturbation.
    !> @param[out] dy The output perturbation.
    subroutine snn_apply_tangent_linear(self, member, dp, dx, dy)
        class(SequentialNeuralNetwork), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        integer(ik) :: i
        if ( self % num_layers == 1 ) then
            call self % list_layers(1) % this_layer % apply_tangent_linear(member, dp, dx, dy)
        else
            call self % list_layers(1) % this_layer % apply_tangent_linear(member,&
                dp(self % ip_start(1):self % ip_end(1)), dx,&
                self % list_layers(2) % this_layer % tangent_linear_input(:, member))
            do i = 2, self % num_layers - 1
                call self % list_layers(i) % this_layer % apply_tangent_linear(member,&
                    dp(self % ip_start(i):self % ip_end(i)),&
                    self % list_layers(i) % this_layer % tangent_linear_input(:, member),&
                    self % list_layers(i+1) % this_layer % tangent_linear_input(:, member))
            end do
            call self % list_layers(self % num_layers) % this_layer % apply_tangent_linear(member,&
                dp(self % ip_start(self % num_layers):self % ip_end(self % num_layers)),&
                self % list_layers(self % num_layers) % this_layer % tangent_linear_input(:, member), dy)
        end if
    end subroutine snn_apply_tangent_linear

    !--------------------------------------------------
    !> @brief Implements \ref sequentialneuralnetwork::apply_adjoint.
    !>
    !> Applies the adjoint of the network.
    !>
    !> @details \b Note
    !>
    !> If there is more than one layer, this method uses
    !> layer::adjoint_input as intermediate storage.
    !>
    !> For this reason, the intent of `self` is declared `inout`.
    !> In addition, the intent of `dy` is declared `intout` because
    !> for certain layer classes (e.g. \ref fnn_layer_dense::denselayer)
    !> the output perturbation gets overwritten in the process. 
    !> @param[inout] self The network.
    !> @param[in] member The index inside the batch.
    !> @param[inout] dy The output perturbation.
    !> @param[out] dp The parameter perturbation.
    !> @param[out] dx The state perturbation.
    subroutine snn_apply_adjoint(self, member, dy, dp, dx)
        class(SequentialNeuralNetwork), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        integer(ik) :: i
        if ( self % num_layers == 1 ) then
            call self % list_layers(1) % this_layer % apply_adjoint(member, dy, dp, dx)
        else
            call self % list_layers(self % num_layers) % this_layer % apply_adjoint(member, dy,&
                dp(self % ip_start(self % num_layers):self % ip_end(self % num_layers)),&
                self % list_layers(self % num_layers-1) % this_layer % adjoint_input(:, member))
            do i = self % num_layers - 1, 2, -1
                call self % list_layers(i) % this_layer % apply_adjoint(member,&
                    self % list_layers(i) % this_layer % adjoint_input(:, member),&
                    dp(self % ip_start(i):self % ip_end(i)),&
                    self % list_layers(i-1) % this_layer % adjoint_input(:, member))
            end do
            call self % list_layers(1) % this_layer % apply_adjoint(member,&
                self % list_layers(1) % this_layer % adjoint_input(:, member),&
                dp(self % ip_start(1):self % ip_end(1)), dx)
        end if
    end subroutine snn_apply_adjoint

end module fnn_network_sequential

