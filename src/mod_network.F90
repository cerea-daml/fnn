
module mod_network

    use mod_kinds, only: ik, rk
    use mod_random
    use mod_layers

    implicit none

    private
    public :: SequentialNeuralNetwork, construct_sequential_neural_network, snn_fromfile

    !--------------------------------------------------
    ! class LayerContainer
    !--------------------------------------------------

    type :: LayerContainer
        class(Layer), allocatable :: this_layer
    end type

    !--------------------------------------------------
    ! class SequentialNeuralNetwork
    !--------------------------------------------------
    !

    type :: SequentialNeuralNetwork
        private
        integer(ik) :: ensemble_size
        integer(ik) :: num_layers
        integer(ik) :: num_parameters
        class(LayerContainer), allocatable :: list_layers(:)
        integer(ik), allocatable :: ip_start(:)
        integer(ik), allocatable :: ip_end(:)
    contains
        procedure, public, pass :: get_input_size => snn_get_input_size
        procedure, public, pass :: get_output_size => snn_get_output_size
        procedure, public, pass :: get_num_parameters => snn_get_num_parameters
        procedure, public, pass :: set_parameters => snn_set_parameters
        procedure, public, pass :: get_parameters => snn_get_parameters
        procedure, public, pass :: tofile => snn_tofile
        procedure, public, pass :: forward => snn_forward
        procedure, public, pass :: tangent_linear => snn_tangent_linear
        procedure, public, pass :: adjoint => snn_adjoint
    end type SequentialNeuralNetwork

contains

    !--------------------------------------------------
    ! methods for class SequentialNeuralNetwork
    !--------------------------------------------------

    type(SequentialNeuralNetwork) function& 
            construct_sequential_neural_network(ensemble_size, internal_sizes, activation_names,&
            initialisation_names) result (self)
        integer(ik), intent(in) :: ensemble_size
        integer(ik), intent(in) :: internal_sizes(:)
        character(len=*), intent(in) :: activation_names(:)
        character(len=*), intent(in) :: initialisation_names(:)
        integer(ik) :: i
        integer(ik) :: ip
        !print *, 'CALLING construct_sequential_neural_network()'
        self % ensemble_size = ensemble_size
        self % num_layers = size(internal_sizes) - 1
        allocate(self % list_layers(self % num_layers))
        allocate(self % ip_start(self % num_layers))
        allocate(self % ip_end(self % num_layers))
        ip = 0
        do i = 1, self % num_layers
            allocate(DenseLayer::self % list_layers(i) % this_layer)
            self % list_layers(i) % this_layer = construct_dense_layer(internal_sizes(i), internal_sizes(i+1),&
                ensemble_size, activation_names(i), initialisation_names(i))
            self % ip_start(i) = ip + 1
            ip = ip + self % list_layers(i) % this_layer % get_num_parameters()
            self % ip_end(i) = ip
        end do
        self % num_parameters = ip
    end function construct_sequential_neural_network

    type(SequentialNeuralNetwork) function snn_fromfile(ensemble_size, filename) result(self)
        integer(ik), intent(in) :: ensemble_size
        character(len=*), intent(in) :: filename
        integer(ik) :: fileunit
        character(len=100) :: network_name
        character(len=100) :: layer_name
        integer(ik) :: i
        integer(ik) :: ip
        !print *, 'CALLING snn_fromfile()'
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
                        self % list_layers(i) % this_layer = norm_layer_fromfile(ensemble_size, fileunit)
                        self % ip_start(i) = ip + 1
                        self % ip_end(i) = ip
                    case default ! default to dense layer
                        allocate(DenseLayer::self % list_layers(i) % this_layer)
                        self % list_layers(i) % this_layer = dense_layer_fromfile(ensemble_size, fileunit)
                        self % ip_start(i) = ip + 1
                        ip = ip + self % list_layers(i) % this_layer % get_num_parameters()
                        self % ip_end(i) = ip
                end select
            end do
            self % num_parameters = ip
        else
            print *, 'ERROR: unknown network name (', trim(network_name), ')'
        end if
    end function snn_fromfile

    integer(ik) function snn_get_input_size(self) result(input_size)
        class(SequentialNeuralNetwork), intent(in) :: self
        !print *, 'CALLING snn_get_input_size()'
        input_size = self % list_layers(1) % this_layer % input_size
    end function snn_get_input_size

    integer(ik) function snn_get_output_size(self) result(output_size)
        class(SequentialNeuralNetwork), intent(in) :: self
        !print *, 'CALLING snn_get_output_size()'
        output_size = self % list_layers(self % num_layers) % this_layer % output_size
    end function snn_get_output_size

    integer(ik) function snn_get_num_parameters(self) result(num_parameters)
        class(SequentialNeuralNetwork), intent(in) :: self
        !print *, 'CALLING snn_get_num_parameters()'
        num_parameters = self % num_parameters
    end function snn_get_num_parameters

    subroutine snn_set_parameters(self, new_parameters)
        class(SequentialNeuralNetwork), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        integer(ik) :: i
        !print *, 'CALLING snn_set_parameters()'
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % set_parameters(&
                new_parameters(self % ip_start(i):self % ip_end(i)))
        end do
    end subroutine snn_set_parameters

    subroutine snn_get_parameters(self, parameters)
        class(SequentialNeuralNetwork), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        integer(ik) :: i
        !print *, 'CALLING snn_get_parameters()'
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % get_parameters(&
                parameters(self % ip_start(i):self % ip_end(i)))
        end do
    end subroutine snn_get_parameters

    subroutine snn_tofile(self, filename)
        class(SequentialNeuralNetwork), intent(in) :: self
        character(len=*), intent(in) :: filename
        integer(ik) :: fileunit
        integer(ik) :: i
        !print *, 'CALLING snn_tofile()'
        open(newunit=fileunit, file=filename, action='write')
        write(fileunit, fmt=*) 'sequential'
        write(fileunit, fmt=*) self % num_layers
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % tofile(fileunit)
        end do
        close(fileunit)
    end subroutine snn_tofile

    subroutine snn_forward(self, member, x, y)
        class(SequentialNeuralNetwork), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        integer(ik) :: i
        !print *, 'CALLING snn_forward()'
        if ( self % num_layers == 1 ) then
            call self % list_layers(1) % this_layer % apply_layer(member, x, y)
        else
            call self % list_layers(1) % this_layer % apply_layer(member, x,&
                self % list_layers(2) % this_layer % apply_in(:, member))
            do i = 2, self % num_layers - 1
                call self % list_layers(i) % this_layer % apply_layer(member,&
                    self % list_layers(i) % this_layer % apply_in(:, member),&
                    self % list_layers(i+1) % this_layer % apply_in(:, member))
            end do
            call self % list_layers(self % num_layers) % this_layer % apply_layer(member,&
                self % list_layers(self % num_layers) % this_layer % apply_in(:, member), y)
        end if
    end subroutine snn_forward

    subroutine snn_tangent_linear(self, member, dp, dx, dy)
        class(SequentialNeuralNetwork), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        integer(ik) :: i
        !print *, 'CALLING snn_tangent_linear()'
        if ( self % num_layers == 1 ) then
            call self % list_layers(1) % this_layer % apply_tangent_linear(member, dp, dx, dy)
        else
            call self % list_layers(1) % this_layer % apply_tangent_linear(member,&
                dp(self % ip_start(1):self % ip_end(1)), dx,&
                self % list_layers(2) % this_layer % tangent_linear_in(:, member))
            do i = 2, self % num_layers - 1
                call self % list_layers(i) % this_layer % apply_tangent_linear(member,&
                    dp(self % ip_start(i):self % ip_end(i)),&
                    self % list_layers(i) % this_layer % tangent_linear_in(:, member),&
                    self % list_layers(i+1) % this_layer % tangent_linear_in(:, member))
            end do
            call self % list_layers(self % num_layers) % this_layer % apply_tangent_linear(member,&
                dp(self % ip_start(self % num_layers):self % ip_end(self % num_layers)),&
                self % list_layers(self % num_layers) % this_layer % tangent_linear_in(:, member), dy)
        end if
    end subroutine snn_tangent_linear

    subroutine snn_adjoint(self, member, dy, dp, dx)
        class(SequentialNeuralNetwork), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        integer(ik) :: i
        !print *, 'CALLING snn_adjoint()'
        if ( self % num_layers == 1 ) then
            call self % list_layers(1) % this_layer % apply_adjoint(member, dy, dp, dx)
        else
            call self % list_layers(self % num_layers) % this_layer % apply_adjoint(member, dy,&
                dp(self % ip_start(self % num_layers):self % ip_end(self % num_layers)),&
                self % list_layers(self % num_layers-1) % this_layer % adjoint_in(:, member))
            do i = self % num_layers - 1, 2, -1
                call self % list_layers(i) % this_layer % apply_adjoint(member,&
                    self % list_layers(i) % this_layer % adjoint_in(:, member),&
                    dp(self % ip_start(i):self % ip_end(i)),&
                    self % list_layers(i-1) % this_layer % adjoint_in(:, member))
            end do
            call self % list_layers(1) % this_layer % apply_adjoint(member,&
                self % list_layers(1) % this_layer % adjoint_in(:, member),&
                dp(self % ip_start(1):self % ip_end(1)), dx)
        end if
    end subroutine snn_adjoint

end module mod_network

