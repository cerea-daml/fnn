
!> @brief Module dedicated to the class \ref frozennormalisationlayer.
module fnn_layer_frozen_normalisation

    use fnn_common
    use fnn_activation_linear
    use fnn_activation_tanh
    use fnn_activation_relu
    use fnn_layer

    implicit none

    private
    public :: FrozenNormalisationLayer, frozen_norm_layer_fromfile

    !--------------------------------------------------
    !> @brief Implements a frozen normalisation layer.
    !> @details This layer has no (trainable) parameters.
    !> It can be used to rescale the input and output
    !> of a network variable per variable.
    type, extends(Layer) :: FrozenNormalisationLayer
        private
        !> The frozen parameters
        real(rk), allocatable :: frozen_parameters(:)
    contains
        !> @brief Reads the parameters from binary file.
        !> Implemented by \ref frozen_norm_read_parameters.
        procedure, pass, public :: read_parameters => frozen_norm_read_parameters
        !> @brief Saves the layer.
        !> Implemented by \ref frozen_norm_tofile.
        procedure, pass, public :: tofile => frozen_norm_tofile
        !> @brief Applies and linearises the layer.
        !> Implemented by \ref frozen_norm_apply_forward.
        procedure, pass, public :: apply_forward => frozen_norm_apply_forward
        !> @brief Applies the TL of the layer.
        !> Implemented by \ref frozen_norm_apply_tangent_linear.
        procedure, pass, public :: apply_tangent_linear => frozen_norm_apply_tangent_linear
        !> @brief Applies the adjoint of the layer.
        !> Implemented by \ref frozen_norm_apply_adjoint.
        procedure, pass, public :: apply_adjoint => frozen_norm_apply_adjoint
    end type FrozenNormalisationLayer

contains

    !--------------------------------------------------
    !> @brief Constructor for class \ref frozennormalisationlayer from a file.
    !> @param[in] batch_size The value for layer::batch_size.
    !> @param[in] unit_num The unit number for the read statements.
    !> @return The constructed layer.
    type(FrozenNormalisationLayer) function frozen_norm_layer_fromfile(batch_size, unit_num) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: unit_num
        read(unit_num, *) self % input_size
        self % output_size = self % input_size
        self % batch_size = batch_size
        self % num_parameters = 0
        allocate(self % frozen_parameters(2 * self % input_size))
        allocate(self % parameters(0))
        allocate(self % forward_input(self % input_size, self % batch_size))
        allocate(self % tangent_linear_input(self % input_size, self % batch_size))
        allocate(self % adjoint_input(self % output_size, self % batch_size))
        self % frozen_parameters = 0
        self % forward_input = 0
        self % tangent_linear_input = 0
        self % adjoint_input = 0
    end function frozen_norm_layer_fromfile

    !--------------------------------------------------
    !> @brief Implements \ref frozennormalisationlayer::read_parameters.
    !>
    !> Reads the parameters from binary file.
    !> @param[inout] self The layer.
    !> @param[in] unit_num The unit number for the read statement.
    subroutine frozen_norm_read_parameters(self, unit_num)
        class(FrozenNormalisationLayer), intent(inout) :: self
        integer(ik), intent(in) :: unit_num
        real(r0), allocatable :: the_parameters(:)
        ! read in r0 precision
        allocate(the_parameters(size(self % frozen_parameters)))
        read(unit_num) the_parameters
        ! cast to rk precision
        self % frozen_parameters = the_parameters
    end subroutine frozen_norm_read_parameters

    !--------------------------------------------------
    !> @brief Implements \ref frozennormalisationlayer::tofile.
    !>
    !> Saves the layer. (Not the parameters)
    !> @param[in] self The layer.
    !> @param[in] unit_num The unit number for the write statement.
    subroutine frozen_norm_tofile(self, unit_num)
        class(FrozenNormalisationLayer), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        write(unit_num, fmt=*) 'frozen-normalisation'
        write(unit_num, fmt=*) self % input_size
    end subroutine frozen_norm_tofile

    !--------------------------------------------------
    !> @brief Implements \ref frozennormalisationlayer::apply_forward.
    !>
    !> Applies and linearises the layer.
    !>
    !> @details The forward function reads
    !> \f[\mathbf{y} = \alpha \mathbf{x} + \beta,\f]
    !> where \f$\alpha\f$ is frozennormalisationlayer::alpha and
    !> \f$\beta\f$ is frozennormalisationlayer::beta.
    !>
    !> \b Note
    !>
    !> Input parameter `member` should be less than layer::batch_size.
    !>
    !> The linearisation is trivial and does not require any operation.
    !> The intent of `self` is declared `inout` instead of `in` because of other
    !> subclasses of \ref fnn_layer::layer.
    !> @param[inout] self The layer.
    !> @param[in] train Whether the model is used in training mode.
    !> @param[in] member The index inside the batch.
    !> @param[in] x The input of the layer.
    !> @param[out] y The output of the layer.
    subroutine frozen_norm_apply_forward(self, train, member, x, y)
        class(FrozenNormalisationLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        y = self % frozen_parameters(1:self % input_size) * x&
            + self % frozen_parameters(self % input_size+1:2*self % input_size)
    end subroutine frozen_norm_apply_forward

    !--------------------------------------------------
    !> @brief Implements \ref frozennormalisationlayer::apply_tangent_linear.
    !>
    !> Applies the TL of the layer.
    !>
    !> @details  The TL operator reads
    !> \f[d\mathbf{y} = \alpha d\mathbf{x}.\f]
    !>
    !> \b Note
    !>
    !> In principle, this method should only be called 
    !> after \ref frozennormalisationlayer::apply_forward.
    !>
    !> Since there is no (trainable) parameters, the
    !> parameter perturbation should be an empty array.
    !> @param[in] self The layer.
    !> @param[in] member The index inside the batch.
    !> @param[in] dp The parameter perturbation.
    !> @param[in] dx The state perturbation.
    !> @param[out] dy The output perturbation.
    subroutine frozen_norm_apply_tangent_linear(self, member, dp, dx, dy)
        class(FrozenNormalisationLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = self % frozen_parameters(1:self % input_size) * dx
    end subroutine frozen_norm_apply_tangent_linear

    !--------------------------------------------------
    !> @brief Implements \ref frozennormalisationlayer::apply_adjoint.
    !>
    !> Applies the adjoint of the layer.
    !>
    !> @details  The adjoint operator reads
    !> \f[d\mathbf{x} = \alpha d\mathbf{y}.\f]
    !>
    !> \b Note
    !>
    !> In principle, this method should only be called after
    !> \ref frozennormalisationlayer::apply_forward.
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
    subroutine frozen_norm_apply_adjoint(self, member, dy, dp, dx)
        class(FrozenNormalisationLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        dx = self % frozen_parameters(1:self % input_size) * dy
    end subroutine frozen_norm_apply_adjoint

end module fnn_layer_frozen_normalisation

