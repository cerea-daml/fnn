
module fnn

    use iso_fortran_env, only: int32, int64, real32, real64, real128

    implicit none

    private
    public :: rk, r0, ik, rand1d, rand2d, NeuralNetwork, nn_fromfile

    integer, parameter :: rk = real64
    integer, parameter :: r0 = real32
    integer, parameter :: ik = int32

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type :: NeuralNetwork
        private
        class(Layer), allocatable :: layer
    contains
        procedure, public, pass :: get_input_size => nn_get_input_size
        procedure, public, pass :: get_output_size => nn_get_output_size
        procedure, public, pass :: get_num_parameters => nn_get_num_parameters
        procedure, public, pass :: set_parameters => nn_set_parameters
        procedure, public, pass :: get_parameters => nn_get_parameters
        procedure, public, pass :: apply_forward => nn_apply_forward
        procedure, public, pass :: apply_tangent_linear => nn_apply_tangent_linear
        procedure, public, pass :: apply_adjoint => nn_apply_adjoint
    end type NeuralNetwork

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type :: LayerContainer
        private
        class(Layer), allocatable :: this_layer
    end type

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type :: Layer
        private
        integer(ik) :: input_size
        integer(ik) :: output_size
        integer(ik) :: batch_size
        integer(ik) :: num_parameters
        real(rk), allocatable :: parameters(:)
        real(rk), allocatable :: forward_input(:, :)
        real(rk), allocatable :: tangent_linear_input(:, :)
        real(rk), allocatable :: adjoint_input(:, :)
    contains
        procedure, pass :: read_parameters => layer_read_parameters
        procedure, pass :: set_parameters => layer_set_parameters
        procedure, pass :: get_parameters => layer_get_parameters
        procedure, pass :: apply_forward => layer_apply_forward
        procedure, pass :: apply_tangent_linear => layer_apply_tangent_linear
        procedure, pass :: apply_adjoint => layer_apply_adjoint
    end type Layer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(Layer) :: SequentialLayer
        private
        integer(ik) :: num_layers
        class(LayerContainer), allocatable :: list_layers(:)
        integer(ik), allocatable :: ip_start(:)
        integer(ik), allocatable :: ip_end(:)
    contains
        procedure, pass :: read_parameters => sequential_read_parameters
        procedure, pass :: set_parameters => sequential_set_parameters
        procedure, pass :: get_parameters => sequential_get_parameters
        procedure, pass :: apply_forward => sequential_apply_forward
        procedure, pass :: apply_tangent_linear => sequential_apply_tangent_linear
        procedure, pass :: apply_adjoint => sequential_apply_adjoint
    end type SequentialLayer
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(Layer) :: LinearLayer
        private
    contains
        procedure, pass :: apply_forward => linear_apply_forward
        procedure, pass :: apply_tangent_linear => linear_apply_tangent_linear
        procedure, pass :: apply_adjoint => linear_apply_adjoint
    end type LinearLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(Layer) :: NormalisationLayer
        private
    contains
        procedure, pass :: apply_forward => normalisation_apply_forward
        procedure, pass :: apply_tangent_linear => normalisation_apply_tangent_linear
        procedure, pass :: apply_adjoint => normalisation_apply_adjoint
    end type NormalisationLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(Layer) :: ActivationLayer
        private
    contains
        procedure, pass :: apply_tangent_linear => activation_apply_tangent_linear
        procedure, pass :: apply_adjoint => activation_apply_adjoint
    end type ActivationLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(ActivationLayer) :: ReluActivationLayer
        private
    contains
        procedure, pass :: apply_forward => relu_activation_apply_forward
    end type ReluActivationLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(ActivationLayer) :: TanhActivationLayer
        private
    contains
        procedure, pass :: apply_forward => tanh_activation_apply_forward
    end type TanhActivationLayer

contains

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine rand1d(x)
        real(rk), intent(out) :: x(:)
        call random_number(x)
    end subroutine rand1d

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine rand2d(x)
        real(rk), intent(out) :: x(:, :)
        call random_number(x)
    end subroutine rand2d

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class NeuralNetwork
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    integer(ik) function nn_get_input_size(self) result(input_size)
        class(NeuralNetwork), intent(in) :: self
        input_size = self % layer % input_size
    end function nn_get_input_size

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    integer(ik) function nn_get_output_size(self) result(output_size)
        class(NeuralNetwork), intent(in) :: self
        output_size = self % layer % output_size
    end function nn_get_output_size

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    integer(ik) function nn_get_num_parameters(self) result(num_parameters)
        class(NeuralNetwork), intent(in) :: self
        num_parameters = self % layer % num_parameters
    end function nn_get_num_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine nn_set_parameters(self, new_parameters)
        class(NeuralNetwork), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        call self % layer % set_parameters(new_parameters)
    end subroutine nn_set_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine nn_get_parameters(self, parameters)
        class(NeuralNetwork), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        call self % layer % get_parameters(parameters)
    end subroutine nn_get_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine nn_apply_forward(self, train, member, x, y)
        class(NeuralNetwork), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        call self % layer % apply_forward(train, member, x, y)
    end subroutine nn_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine nn_apply_tangent_linear(self, member, dp, dx, dy)
        class(NeuralNetwork), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        call self % layer % apply_tangent_linear(member, dp, dx, dy)
    end subroutine nn_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine nn_apply_adjoint(self, member, dy, dp, dx)
        class(NeuralNetwork), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        call self % layer % apply_adjoint(member, dy, dp, dx)
    end subroutine nn_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class Layer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_read_parameters(self, fileunit)
        class(Layer), intent(inout) :: self
        integer(ik), intent(in) :: fileunit
        real(r0), allocatable :: the_parameters(:)
        ! read in r0 precision
        allocate(the_parameters(size(self % parameters)))
        read(fileunit) the_parameters
        ! cast to rk precision
        self % parameters = the_parameters
    end subroutine layer_read_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_set_parameters(self, new_parameters)
        class(Layer), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        self % parameters = new_parameters
    end subroutine layer_set_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_get_parameters(self, parameters)
        class(Layer), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        parameters = self % parameters
    end subroutine layer_get_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_apply_forward(self, train, member, x, y)
        class(Layer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        print *, 'WARNING: using non-implemented method Layer::apply_forward()'
    end subroutine layer_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_apply_tangent_linear(self, member, dp, dx, dy)
        class(Layer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        print *, 'WARNING: using non-implemented method Layer::apply_tangent_linear()'
    end subroutine layer_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_apply_adjoint(self, member, dy, dp, dx)
        class(Layer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        print *, 'WARNING: using non-implemented method Layer::apply_adjoint()'
    end subroutine layer_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class SequentialLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_read_parameters(self, fileunit)
        class(SequentialLayer), intent(inout) :: self
        integer(ik), intent(in) :: fileunit
        integer(ik) :: i
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % read_parameters(fileunit)
        end do
    end subroutine sequential_read_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_set_parameters(self, new_parameters)
        class(SequentialLayer), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        integer(ik) :: i
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % set_parameters(&
                new_parameters(self % ip_start(i):self % ip_end(i)))
        end do
    end subroutine sequential_set_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_get_parameters(self, parameters)
        class(SequentialLayer), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        integer(ik) :: i
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % get_parameters(&
                parameters(self % ip_start(i):self % ip_end(i)))
        end do
    end subroutine sequential_get_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_apply_forward(self, train, member, x, y)
        class(SequentialLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        integer(ik) :: i
        ! this may not be necessary if we use the forward input from the first layer...
        self % forward_input(:, member) = x
        if ( self % num_layers == 1 ) then
            call self % list_layers(1) % this_layer % apply_forward(&
                train, member, x, y)
        else
            call self % list_layers(1) % this_layer % apply_forward(&
                train, member, x,&
                self % list_layers(2) % this_layer % forward_input(:, member))
            do i = 2, self % num_layers - 1
                call self % list_layers(i) % this_layer % apply_forward(&
                    train, member,&
                    self % list_layers(i) % this_layer % forward_input(:, member),&
                    self % list_layers(i+1) % this_layer % forward_input(:, member))
            end do
            call self % list_layers(self % num_layers) % this_layer % apply_forward(&
                train, member,&
                self % list_layers(self % num_layers) % this_layer % forward_input(:, member), y)
        end if
    end subroutine sequential_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_apply_tangent_linear(self, member, dp, dx, dy)
        class(SequentialLayer), intent(inout) :: self
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
    end subroutine sequential_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_apply_adjoint(self, member, dy, dp, dx)
        class(sequentialLayer), intent(inout) :: self
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
    end subroutine sequential_apply_adjoint
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class LinearLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine linear_apply_forward(self, train, member, x, y)
        class(LinearLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        self % forward_input(:, member) = x
        y = matmul(&
            reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            x)
        y = y + self % parameters(1:self % output_size)
    end subroutine linear_apply_forward
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine linear_apply_tangent_linear(self, member, dp, dx, dy)
        class(LinearLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = matmul(&
            reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            dx)
        if ( self % num_parameters > 0 ) then
            dy = dy + matmul(&
                reshape(dp(self % output_size+1:self % output_size*(self % input_size+1)),&
                [self % output_size, self % input_size]),&
                self % forward_input(:, member))
            dy = dy + dp(1:self % output_size)
        end if
    end subroutine linear_apply_tangent_linear
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine linear_apply_adjoint(self, member, dy, dp, dx)
        class(LinearLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        if ( self % num_parameters > 0 ) then
            dp(1:self % output_size) = dy
            dp(self % output_size+1:self % output_size*(self % input_size+1)) = reshape(matmul(&
                reshape(dp(1:self % output_size), [self % output_size, 1]),&
                reshape(self % forward_input(:, member), [1, self % input_size])),&
                [self % output_size*self % input_size])
        end if
        dx = matmul(transpose(reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size])), dy)
    end subroutine linear_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class NormalisationLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine normalisation_apply_forward(self, train, member, x, y)
        class(NormalisationLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        self % forward_input(:, member) = x
        y = self % parameters(1:self % input_size) * x&
            + self % parameters(self % input_size+1:2*self % input_size)
    end subroutine normalisation_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine normalisation_apply_tangent_linear(self, member, dp, dx, dy)
        class(NormalisationLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = self % parameters(1:self % input_size) * dx
        if ( self % num_parameters > 0 ) then
             dy = dy + dp(1:self % input_size) * self % forward_input(:, member)&
                  + dp(self % input_size+1:2*self % input_size)
        end if
    end subroutine normalisation_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine normalisation_apply_adjoint(self, member, dy, dp, dx)
        class(NormalisationLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        dx = self % parameters(1:self % input_size) * dy
        if ( self % num_parameters > 0 ) then
            dp(1:self % input_size) = self % forward_input(:, member) * dy
            dp(self % input_size+1:2*self % input_size) = dy
        end if
    end subroutine normalisation_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class ActivationLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine activation_apply_tangent_linear(self, member, dp, dx, dy)
        class(ActivationLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = self % forward_input(:, member) * dx
    end subroutine activation_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine activation_apply_adjoint(self, member, dy, dp, dx)
        class(ActivationLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        dx = self % forward_input(:, member) * dy
    end subroutine activation_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class ReluActivationLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine relu_activation_apply_forward(self, train, member, x, y)
        class(ReluActivationLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        integer(ik) :: i
        do i = 1, size(y)
            if (x(i) > 0) then
                y(i) = x(i)
                self % forward_input(i, member) = 1
            else
                y(i) = 0
                self % forward_input(i, member) = 0
            end if
        end do
    end subroutine relu_activation_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class TanhActivationLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine tanh_activation_apply_forward(self, train, member, x, y)
        class(TanhActivationLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        y = tanh(x)
        self % forward_input(:, member) = 1 - y**2
    end subroutine tanh_activation_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! constructors
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(NeuralNetwork) function nn_fromfile(batch_size, filename_txt, filename_bin) result(self)
        integer(ik), intent(in) :: batch_size
        character(len=*), intent(in) :: filename_txt
        character(len=*), intent(in) :: filename_bin
        integer(ik) :: fileunit
        character(len=100) :: layer_name
        ! read architecture
        open(newunit=fileunit, file=filename_txt, action='read')
        read(fileunit, fmt=*) layer_name
        select case(trim(layer_name)) ! loop over all layers here
            case('linear')
                allocate(LinearLayer::self % layer)
                self % layer = linear_layer_fromfile(batch_size, fileunit, .false.)
            case('frozen_linear')
                allocate(LinearLayer::self % layer)
                self % layer = linear_layer_fromfile(batch_size, fileunit, .true.)
            case('normalisation')
                allocate(NormalisationLayer::self % layer)
                self % layer = normalisation_layer_fromfile(batch_size, fileunit, .false.)
            case('frozen_normalisation')
                allocate(NormalisationLayer::self % layer)
                self % layer = normalisation_layer_fromfile(batch_size, fileunit, .true.)
            case('relu_activation')
                allocate(ReluActivationLayer::self % layer)
                self % layer = relu_activation_layer_fromfile(batch_size, fileunit)
            case('tanh_activation')
                allocate(TanhActivationLayer::self % layer)
                self % layer = tanh_activation_layer_fromfile(batch_size, fileunit)
            case default
                allocate(SequentialLayer::self % layer)
                self % layer = sequential_layer_fromfile(batch_size, fileunit)
        end select
        close(fileunit)
        ! read parameters
        open(newunit=fileunit, file=filename_bin, form='unformatted', access='stream', action='read')
        call self % layer % read_parameters(fileunit)
        close(fileunit)
    end function nn_fromfile
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(SequentialLayer) recursive function sequential_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        character(len=100) :: layer_name
        integer(ik) :: i
        integer(ik) :: ip
        read(fileunit, *) self % num_layers
        allocate(self % list_layers(self % num_layers))
        allocate(self % ip_start(self % num_layers))
        allocate(self % ip_end(self % num_layers))
        ip = 0
        do i = 1, self % num_layers
            read(fileunit, *) layer_name
            select case(trim(layer_name)) ! loop over all layers here
                case('linear')
                    allocate(LinearLayer::self % list_layers(i) % this_layer)
                    self % list_layers(i) % this_layer = linear_layer_fromfile(batch_size, fileunit, .false.)
                case('frozen_linear')
                    allocate(LinearLayer::self % list_layers(i) % this_layer)
                    self % list_layers(i) % this_layer = linear_layer_fromfile(batch_size, fileunit, .true.)
                case('normalisation')
                    allocate(NormalisationLayer::self % list_layers(i) % this_layer)
                    self % list_layers(i) % this_layer = normalisation_layer_fromfile(batch_size, fileunit, .false.)
                case('frozen_normalisation')
                    allocate(NormalisationLayer::self % list_layers(i) % this_layer)
                    self % list_layers(i) % this_layer = normalisation_layer_fromfile(batch_size, fileunit, .true.)
                case('relu_activation')
                    allocate(ReluActivationLayer::self % list_layers(i) % this_layer)
                    self % list_layers(i) % this_layer = relu_activation_layer_fromfile(batch_size, fileunit)
                case('tanh_activation')
                    allocate(TanhActivationLayer::self % list_layers(i) % this_layer)
                    self % list_layers(i) % this_layer = tanh_activation_layer_fromfile(batch_size, fileunit)
                case default
                    allocate(SequentialLayer::self % list_layers(i) % this_layer)
                    self % list_layers(i) % this_layer = sequential_layer_fromfile(batch_size, fileunit)
            end select
            self % ip_start(i) = ip + 1
            ip = ip + self % list_layers(i) % this_layer % num_parameters
            self % ip_end(i) = ip
        end do
        self % input_size = self % list_layers(1) % this_layer % input_size
        self % output_size = self % list_layers(self % num_layers) % this_layer % output_size
        self % batch_size = batch_size
        self % num_parameters = ip
        allocate(self % parameters(0))
        ! Q: should we use self % list_layers(1) % this_layer % forward_input as self % forward_input ???
        allocate(self % forward_input(self % input_size, self % batch_size))
        ! Q: should we use self % list_layers(1) % this_layer % tangent_linear_input as self % tangent_linear_input ???
        allocate(self % tangent_linear_input(self % input_size, self % batch_size))
        ! Q: should we use self % list_layers(self % num_layers) % this_layer % adjoint_input as self % adjoint_input ???
        allocate(self % adjoint_input(self % output_size, self % batch_size))
        self % parameters = 0
        self % forward_input = 0
        self % tangent_linear_input = 0
        self % adjoint_input = 0
    end function sequential_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(LinearLayer) function linear_layer_fromfile(batch_size, fileunit, frozen) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        logical, intent(in) :: frozen
        read(fileunit, *) self % input_size
        read(fileunit, *) self % output_size
        self % batch_size = batch_size
        self % num_parameters = (self % input_size+1) * self % output_size
        allocate(self % parameters(self % num_parameters))
        allocate(self % forward_input(self % input_size, self % batch_size))
        allocate(self % tangent_linear_input(self % input_size, self % batch_size))
        allocate(self % adjoint_input(self % output_size, self % batch_size))
        self % parameters = 0
        self % forward_input = 0
        self % tangent_linear_input = 0
        self % adjoint_input = 0
        if ( frozen ) then
            self % num_parameters = 0
        end if
    end function linear_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(NormalisationLayer) function normalisation_layer_fromfile(batch_size, fileunit, frozen) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        logical, intent(in) :: frozen
        read(fileunit, *) self % input_size
        self % output_size = self % input_size
        self % batch_size = batch_size
        self % num_parameters = 2 * self % input_size
        allocate(self % parameters(self % num_parameters))
        allocate(self % forward_input(self % input_size, self % batch_size))
        allocate(self % tangent_linear_input(self % input_size, self % batch_size))
        allocate(self % adjoint_input(self % output_size, self % batch_size))
        self % parameters = 0
        self % forward_input = 0
        self % tangent_linear_input = 0
        self % adjoint_input = 0
        if ( frozen ) then
            self % num_parameters = 0
        end if
    end function normalisation_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(ActivationLayer) function activation_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        read(fileunit, *) self % input_size
        self % output_size = self % input_size
        self % batch_size = batch_size
        self % num_parameters = 0
        allocate(self % parameters(0))
        allocate(self % forward_input(self % input_size, self % batch_size))
        allocate(self % tangent_linear_input(self % input_size, self % batch_size))
        allocate(self % adjoint_input(self % output_size, self % batch_size))
        self % parameters = 0
        self % forward_input = 0
        self % tangent_linear_input = 0
        self % adjoint_input = 0
    end function activation_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(ReluActivationLayer) function relu_activation_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        self % ActivationLayer = activation_layer_fromfile(batch_size, fileunit)
    end function relu_activation_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(TanhActivationLayer) function tanh_activation_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        self % ActivationLayer = activation_layer_fromfile(batch_size, fileunit)
    end function tanh_activation_layer_fromfile

end module fnn
