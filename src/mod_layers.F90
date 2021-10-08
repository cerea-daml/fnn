
module mod_layers

    use mod_kinds, only: ik, rk
    use mod_random
    use mod_activation

    implicit none

    private
    public :: Layer, DenseLayer, NormalisationLayer, construct_dense_layer, dense_layer_fromfile, norm_layer_fromfile

    !--------------------------------------------------
    ! class Layer
    !--------------------------------------------------
    !
    ! Base class for all layers. Do not instanciate
    !
    ! Attributes
    ! ----------
    ! input_size : int
    !     The dimension of the input of the layer.
    ! output_size : int
    !     The dimension of the output of the layer.
    ! ensemble_size : int
    !     The number of simulatneous linearisations to compute.
    ! activation : LinearActivation
    !     The activation function.
    !

    type :: Layer
        integer(ik) :: input_size
        integer(ik) :: output_size
        integer(ik) :: ensemble_size
        class(LinearActivation), allocatable :: activation
        real(rk), allocatable :: parameters(:)
        real(rk), allocatable :: apply_in(:, :)
        real(rk), allocatable :: tangent_linear_in(:, :)
        real(rk), allocatable :: adjoint_in(:, :)
    contains
        procedure, public, pass :: get_num_parameters => layer_get_num_parameters
        procedure, public, pass :: set_parameters => layer_set_parameters
        procedure, public, pass :: get_parameters => layer_get_parameters
        procedure, public, pass :: tofile => layer_tofile
        procedure, public, pass :: apply_layer => layer_apply_layer
        procedure, public, pass :: apply_tangent_linear => layer_apply_tangent_linear
        procedure, public, pass :: apply_adjoint => layer_apply_adjoint
    end type Layer

    !--------------------------------------------------
    ! class DenseLayer
    !--------------------------------------------------
    !
    ! Extends Layer.
    ! Implements a dense (fully-connected) layer.
    !
    ! Attributes
    ! ----------
    ! parameters : 1d array of reals
    !     The array containing the parameters of the layer.
    !

    type, extends(Layer) :: DenseLayer
    contains
        procedure, public, pass :: tofile => dense_tofile
        procedure, public, pass :: apply_layer => dense_apply_layer
        procedure, public, pass :: apply_tangent_linear => dense_apply_tangent_linear
        procedure, public, pass :: apply_adjoint => dense_apply_adjoint
    end type DenseLayer

    !--------------------------------------------------
    ! class NormalisationLayer
    !--------------------------------------------------

    type, extends(Layer) :: NormalisationLayer
        real(rk) :: alpha
        real(rk) :: beta
    contains
        procedure, public, pass :: tofile => norm_tofile
        procedure, public, pass :: apply_layer => norm_apply_layer
        procedure, public, pass :: apply_tangent_linear => norm_apply_tangent_linear
        procedure, public, pass :: apply_adjoint => norm_apply_adjoint
    end type NormalisationLayer

contains

    !--------------------------------------------------
    ! methods for class Layer
    !--------------------------------------------------

    integer(ik) function layer_get_num_parameters(self) result(num_parameters)
        class(Layer), intent(in) :: self
        !print *, 'CALLING layer_get_num_parameters()'
        num_parameters = size(self % parameters)
    end function layer_get_num_parameters

    subroutine layer_set_parameters(self, new_parameters)
        class(Layer), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        !print *, 'CALLING layer_set_parameters()'
        self % parameters = new_parameters
    end subroutine layer_set_parameters

    subroutine layer_get_parameters(self, parameters)
        class(Layer), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        !print *, 'CALLING layer_get_parameters()'
        parameters = self % parameters
    end subroutine layer_get_parameters

    subroutine layer_tofile(self, unit_num)
        class(Layer), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        !print *, 'CALLING layer_tofile()'
    end subroutine layer_tofile

    subroutine layer_apply_layer(self, member, x, y)
        class(Layer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        !print *, 'CALLING layer_apply_layer()'
    end subroutine layer_apply_layer

    subroutine layer_apply_tangent_linear(self, member, dp, dx, dy)
        class(Layer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        !print *, 'CALLING layer_apply_tangent_linear()'
    end subroutine layer_apply_tangent_linear

    subroutine layer_apply_adjoint(self, member, dy, dp, dx)
        class(Layer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        !print *, 'CALLING layer_apply_adjoint()'
    end subroutine layer_apply_adjoint

    !--------------------------------------------------
    ! methods for class DenseLayer
    !--------------------------------------------------

    type(DenseLayer) function construct_dense_layer(input_size, output_size,&
            ensemble_size, activation_name, initialisation_name) result(self)
        integer(ik), intent(in) :: input_size
        integer(ik), intent(in) :: output_size
        integer(ik), intent(in) :: ensemble_size
        character(len=*), intent(in) :: activation_name
        character(len=*), intent(in) :: initialisation_name
        !print *, 'CALLING construct_dense_layer()'
        self % input_size = input_size
        self % output_size = output_size
        self % ensemble_size = ensemble_size
        select case(trim(activation_name))
            case('tanh')
                allocate(TanhActivation::self % activation)
                self % activation = construct_tanh_activation(output_size, ensemble_size)
            case default
                allocate(LinearActivation::self % activation)
                self % activation = construct_linear_activation(output_size, ensemble_size)
        end select
        allocate(self % parameters((input_size+1)*output_size))
        select case(trim(initialisation_name))
            case('rand')
                call rand(self % parameters)
            case default
                self % parameters = 0
        end select
        allocate(self % apply_in(input_size, ensemble_size))
        allocate(self % tangent_linear_in(input_size, ensemble_size))
        allocate(self % adjoint_in(output_size, ensemble_size))
    end function construct_dense_layer

    type(DenseLayer) function dense_layer_fromfile(ensemble_size, unit_num) result (self)
        integer(ik), intent(in) :: ensemble_size
        integer(ik), intent(in) :: unit_num
        character(len=100) :: activation_name
        read(unit_num, *) self % input_size
        read(unit_num, *) self % output_size
        self % ensemble_size = ensemble_size
        allocate(self % parameters((self % input_size+1)*self % output_size))
        read(unit_num, *) self % parameters
        read(unit_num, *) activation_name
        select case(trim(activation_name))
            case('tanh')
                allocate(TanhActivation::self % activation)
                self % activation = construct_tanh_activation(self % output_size, self % ensemble_size)
            case default
                allocate(LinearActivation::self % activation)
                self % activation = construct_linear_activation(self % output_size, self % ensemble_size)
        end select
        allocate(self % apply_in(self % input_size, self % ensemble_size))
        allocate(self % tangent_linear_in(self % input_size, self % ensemble_size))
        allocate(self % adjoint_in(self % output_size, self % ensemble_size))
    end function dense_layer_fromfile

    subroutine dense_tofile(self, unit_num)
        class(DenseLayer), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        write(unit_num, fmt=*) 'dense'
        write(unit_num, fmt=*) self % input_size
        write(unit_num, fmt=*) self % output_size
        write(unit_num, fmt=*) self % parameters
        call self % activation % tofile(unit_num)
    end subroutine dense_tofile

    subroutine dense_apply_layer(self, member, x, y)
        class(DenseLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        !print *, 'CALLING dense_apply_layer()'
        ! Question: What happens if we call this function with x=self % apply_in(:, member)???
        ! Does it make an unecessary copy?
        self % apply_in(:, member) = x
        ! Question: is there a way to store internally a view to the kernel and the bias???
        y = matmul(&
            reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            x)
        y = y + self % parameters(1:self % output_size)
        call self % activation % apply_activation(member, y, y)
    end subroutine dense_apply_layer

    subroutine dense_apply_tangent_linear(self, member, dp, dx, dy)
        class(DenseLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        !print *, 'CALLING dense_apply_tangent_linear()'
        dy = matmul(&
            reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            dx)
        dy = dy + matmul(&
            reshape(dp(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            self % apply_in(:, member))
        dy = dy + dp(1:self % output_size)
        call self % activation % apply_tangent_linear(member, dy, dy)
    end subroutine dense_apply_tangent_linear

    subroutine dense_apply_adjoint(self, member, dy, dp, dx)
        class(DenseLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        !print *, 'CALLING dense_apply_adjoint()'
        call self % activation % apply_adjoint(member, dy, dp(1:self % output_size))
        dp(self % output_size+1:self % output_size*(self % input_size+1)) = reshape(matmul(&
            reshape(dp(1:self % output_size), [self % output_size, 1]),&
            reshape(self % apply_in(:, member), [1, self % input_size])),&
            [self % output_size*self % input_size])
        dx = matmul(transpose(reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size])), dp(1:self % output_size))
    end subroutine dense_apply_adjoint

    !--------------------------------------------------
    ! methods for class NormalisationLayer
    !--------------------------------------------------

    type(NormalisationLayer) function norm_layer_fromfile(ensemble_size, unit_num) result (self)
        integer(ik), intent(in) :: ensemble_size
        integer(ik), intent(in) :: unit_num
        read(unit_num, *) self % input_size
        read(unit_num, *) self % alpha
        read(unit_num, *) self % beta
        self % output_size = self % input_size
        self % ensemble_size = ensemble_size
        allocate(self % parameters(0))
        allocate(self % apply_in(self % input_size, self % ensemble_size))
        allocate(self % tangent_linear_in(self % input_size, self % ensemble_size))
        allocate(self % adjoint_in(self % output_size, self % ensemble_size))
    end function norm_layer_fromfile

    subroutine norm_tofile(self, unit_num)
        class(NormalisationLayer), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        write(unit_num, fmt=*) 'normalisation'
        write(unit_num, fmt=*) self % input_size
        write(unit_num, fmt=*) self % alpha
        write(unit_num, fmt=*) self % beta
        call self % activation % tofile(unit_num)
    end subroutine norm_tofile

    subroutine norm_apply_layer(self, member, x, y)
        class(NormalisationLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        y = self % alpha * x + self % beta
    end subroutine norm_apply_layer

    subroutine norm_apply_tangent_linear(self, member, dp, dx, dy)
        class(NormalisationLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = self % alpha * dx
    end subroutine norm_apply_tangent_linear

    subroutine norm_apply_adjoint(self, member, dy, dp, dx)
        class(NormalisationLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        dx = self % alpha * dy
    end subroutine norm_apply_adjoint

end module mod_layers

