
!> @brief Module dedicated to the class \ref normalisationlayer.
module fnn_layer_normalisation

    use fnn_common
    use fnn_activation_linear
    use fnn_activation_tanh
    use fnn_activation_relu
    use fnn_layer

    implicit none

    private
    public :: NormalisationLayer, norm_layer_fromfile

    !--------------------------------------------------
    !> @brief Implements a normalisation layer.
    !> @details This layer has no (trainable) parameters.
    !> It can be used to rescale the input and output
    !> of a network variable per variable.
    type, extends(Layer) :: NormalisationLayer
        private
    contains
        !> @brief Reads the parameters from binary file.
        !> Implemented by \ref norm_read_parameters.
        procedure, pass, public :: read_parameters => norm_read_parameters
        !> @brief Saves the layer.
        !> Implemented by \ref norm_tofile.
        procedure, pass, public :: tofile => norm_tofile
        !> @brief Applies and linearises the layer.
        !> Implemented by \ref norm_apply_forward.
        procedure, pass, public :: apply_forward => norm_apply_forward
        !> @brief Applies the TL of the layer.
        !> Implemented by \ref norm_apply_tangent_linear.
        procedure, pass, public :: apply_tangent_linear => norm_apply_tangent_linear
        !> @brief Applies the adjoint of the layer.
        !> Implemented by \ref norm_apply_adjoint.
        procedure, pass, public :: apply_adjoint => norm_apply_adjoint
    end type NormalisationLayer

contains

    !--------------------------------------------------
    !> @brief Constructor for class \ref normalisationlayer from a file.
    !> @param[in] batch_size The value for layer::batch_size.
    !> @param[in] unit_num The unit number for the read statements.
    !> @return The constructed layer.
    type(NormalisationLayer) function norm_layer_fromfile(batch_size, unit_num) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: unit_num
        read(unit_num, *) self % input_size
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
    end function norm_layer_fromfile

    !--------------------------------------------------
    !> @brief Implements \ref normalisationlayer::read_parameters.
    !>
    !> Reads the parameters from binary file.
    !> @param[inout] self The layer.
    !> @param[in] unit_num The unit number for the read statement.
    subroutine norm_read_parameters(self, unit_num)
        class(NormalisationLayer), intent(inout) :: self
        integer(ik), intent(in) :: unit_num
        real(r0), allocatable :: the_parameters(:)
        ! read in r0 precision
        allocate(the_parameters(size(self % parameters)))
        read(unit_num) the_parameters
        ! cast to rk precision
        self % parameters = the_parameters
    end subroutine norm_read_parameters

    !--------------------------------------------------
    !> @brief Implements \ref normalisationlayer::tofile.
    !>
    !> Saves the layer. (Not the parameters)
    !> @param[in] self The layer.
    !> @param[in] unit_num The unit number for the write statement.
    subroutine norm_tofile(self, unit_num)
        class(NormalisationLayer), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        write(unit_num, fmt=*) 'normalisation'
        write(unit_num, fmt=*) self % input_size
    end subroutine norm_tofile

    !--------------------------------------------------
    !> @brief Implements \ref normalisationlayer::apply_forward.
    !>
    !> Applies and linearises the layer.
    !>
    !> @details The forward function reads
    !> \f[\mathbf{y} = \alpha \mathbf{x} + \beta,\f]
    !> where \f$\alpha\f$ is normalisationlayer::alpha and
    !> \f$\beta\f$ is normalisationlayer::beta.
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
    subroutine norm_apply_forward(self, train, member, x, y)
        class(NormalisationLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        y = self % parameters(1:self % input_size) * x&
            + self % parameters(self % input_size+1:2*self % input_size)
    end subroutine norm_apply_forward

    !--------------------------------------------------
    !> @brief Implements \ref normalisationlayer::apply_tangent_linear.
    !>
    !> Applies the TL of the layer.
    !>
    !> @details  The TL operator reads
    !> \f[d\mathbf{y} = \alpha d\mathbf{x} + d\alpha\matbf{x}
    !> + d\beta.\f]
    !>
    !> \b Note
    !>
    !> In principle, this method should only be called 
    !> after \ref normalisationlayer::apply_forward.
    !>
    !> Since there is no (trainable) parameters, the
    !> parameter perturbation should be an empty array.
    !> @param[in] self The layer.
    !> @param[in] member The index inside the batch.
    !> @param[in] dp The parameter perturbation.
    !> @param[in] dx The state perturbation.
    !> @param[out] dy The output perturbation.
    subroutine norm_apply_tangent_linear(self, member, dp, dx, dy)
        class(NormalisationLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = self % parameters(1:self % input_size) * dx&
             + dp(1:self % input_size) * self % forward_input(:, member)&
             + dp(self % input_size+1:2*self % input_size)
    end subroutine norm_apply_tangent_linear

    !--------------------------------------------------
    !> @brief Implements \ref normalisationlayer::apply_adjoint.
    !>
    !> Applies the adjoint of the layer.
    !>
    !> @details  The adjoint operator reads
    !> \f[d\mathbf{x} = \alpha d\mathbf{y}.\f]
    !> \f[d\alpha = \mathbf{x} d\mathbf{y}.\f]
    !> \f[d\beta = d\mathbf{y}.\f]
    !>
    !> \b Note
    !>
    !> In principle, this method should only be called after
    !> \ref normalisationlayer::apply_forward.
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
    subroutine norm_apply_adjoint(self, member, dy, dp, dx)
        class(NormalisationLayer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        dx = self % parameters(1:self % input_size) * dy
        dp(1:self % input_size) = self % forward_input(:, member) * dy
        dp(self % input_size+1:2*self % input_size) = dy
    end subroutine norm_apply_adjoint

end module fnn_layer_normalisation

