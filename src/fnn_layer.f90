
!> @brief Module dedicated to the class \ref layer.
module fnn_layer

    use fnn_common
    use fnn_activation_linear

    implicit none

    private
    public :: Layer

    !--------------------------------------------------
    !> @brief Base class for all layers.
    !> Do not instanciate.
    !> @details This class only exists to enable polymorphism.
    !>
    !> \b Note
    !>
    !> All attributes are public because they need
    !> to be accessed in subclasses and in networks.
    !> This could be avoided if all layers and networks
    !> would be part of the same module.
    type :: Layer
        private
        !> The dimension of the input of the layer.
        integer(ik), public :: input_size
        !> The dimension of the output of the layer.
        integer(ik), public :: output_size
        !> The batch size.
        integer(ik), public :: batch_size
        !> The number of (trainable) parameters.
        integer(ik), public :: num_parameters
        !> The activation function.
        class(LinearActivation), allocatable, public :: activation
        !> @brief The vector of (trainable) parameters.
        !> @details All trainable parameters are stored into a single
        !> array of reals with shape (layer::num_parameters).
        real(rk), allocatable, public :: parameters(:)
        !> @brief The storage for the layer::apply_forward method.
        !> @details Array of reals with shape (layer::input_size, layer::batch_size).
        !>
        !> \b Note
        !>
        !> This array can be used by network classes to apply
        !> successive layers. In addition, this array is used to
        !> store the linearisation.
        real(rk), allocatable, public :: forward_input(:, :)
        !> @brief The storage for the layer::apply_tangent_linear method.
        !> @details Array of reals with shape (layer::input_size, layer::batch_size).
        !>
        !> \b Note
        !>
        !> This array can be used by network classes to apply successive layers.
        real(rk), allocatable, public :: tangent_linear_input(:, :)
        !> @brief The storage for the layer::apply_adjoint method.
        !> @details Array of reals with shape (layer::output_size, layer::batch_size).
        !>
        !> \b Note
        !>
        !> This array can be used by network classes to apply successive layers.
        real(rk), allocatable, public :: adjoint_input(:, :)
    contains
        !> @brief Returns the number of parameters.
        !> Implemented by \ref layer_get_num_parameters.
        procedure, pass, public :: get_num_parameters => layer_get_num_parameters
        !> @brief Setter for layer::parameters.
        !> Implemented by \ref layer_set_parameters.
        procedure, pass, public :: set_parameters => layer_set_parameters
        !> @brief Getter for layer::parameters.
        !> Implemented by \ref layer_get_parameters.
        procedure, pass, public :: get_parameters => layer_get_parameters
        !> @brief Saves the layer.
        !> Implemented by \ref layer_tofile.
        procedure, pass, public :: tofile => layer_tofile
        !> @brief Applies and linearises the layer.
        !> Implemented by \ref layer_apply_forward.
        procedure, pass, public :: apply_forward => layer_apply_forward
        !> @brief Applies the TL of the layer.
        !> Implemented by \ref layer_apply_tangent_linear.
        procedure, pass, public :: apply_tangent_linear => layer_apply_tangent_linear
        !> @brief Applies the adjoint of the layer.
        !> Implemented by \ref layer_apply_adjoint.
        procedure, pass, public :: apply_adjoint => layer_apply_adjoint
    end type Layer

contains

    !--------------------------------------------------
    !> @brief Implements \ref layer::get_num_parameters.
    !>
    !> Returns the number of parameters.
    !>
    !> @param[in] self The layer.
    !> @return The number of parameters.
    integer(ik) function layer_get_num_parameters(self) result(num_parameters)
        class(Layer), intent(in) :: self
        num_parameters = self % num_parameters
    end function layer_get_num_parameters

    !--------------------------------------------------
    !> @brief Implements \ref layer::set_parameters.
    !>
    !> Setter for layer::parameters.
    !>
    !> @param[inout] self The layer.
    !> @param[in] new_parameters The new values for the parameters.
    subroutine layer_set_parameters(self, new_parameters)
        class(Layer), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        self % parameters = new_parameters
    end subroutine layer_set_parameters

    !--------------------------------------------------
    !> @brief Implements \ref layer::get_parameters.
    !>
    !> Getter for layer::parameters.
    !>
    !> @param[in] self The layer.
    !> @param[out] parameters The vector of parameters.
    subroutine layer_get_parameters(self, parameters)
        class(Layer), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        parameters = self % parameters
    end subroutine layer_get_parameters

    !--------------------------------------------------
    !> @brief Implements \ref layer::tofile.
    !>
    !> Saves the layer.
    !> @details \b Note
    !> 
    !> This should be overridden by each subclass.
    !> @param[in] self The layer.
    !> @param[in] unit_num The unit number for the write statement.
    subroutine layer_tofile(self, unit_num)
        class(Layer), intent(in) :: self
        integer(ik), intent(in) :: unit_num
        print *, 'WARNING: using non-implemented method Layer::tofile()'
    end subroutine layer_tofile

    !--------------------------------------------------
    !> @brief Implements \ref layer::apply_forward.
    !>
    !> Applies and linearises the layer.
    !>
    !> @details The forward function reads
    !> \f[ \mathbf{y} = \mathcal{F}(\mathbf{p}, \mathbf{x}),\f]
    !> where \f$\mathbf{p}\f$ is the vector of parameters.
    !>
    !> \b Note
    !>
    !> This should be overridden by each subclass.
    !>
    !> The intent for `self` is declared `inout` instead of `in` because,
    !> for certain subclasses (e.g. \ref fnn_layer_dense::denselayer)
    !> the linearisation is stored inside the layer.
    !> @param[inout] self The layer.
    !> @param[in] train Whether the model is used in training mode.
    !> @param[in] member The index inside the batch.
    !> @param[in] x The input of the layer.
    !> @param[out] y The output of the layer.
    subroutine layer_apply_forward(self, train, member, x, y)
        class(Layer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        print *, 'WARNING: using non-implemented method Layer::apply_forward()'
    end subroutine layer_apply_forward

    !--------------------------------------------------
    !> @brief Implements \ref layer::apply_tangent_linear.
    !>
    !> Applies the TL of the layer.
    !>
    !> @details The TL operator reads
    !> \f[ d\mathbf{y} = \mathbf{F}^\mathrm{p}d\mathbf{p} + 
    !> \mathbf{F}^\mathrm{x}d\mathbf{x},\f]
    !> where \f$\mathbf{F}^\mathrm{p}\f$ is the TL of \f$\mathcal{F}\f$
    !> with respect to the \f$\mathbf{p}\f$ component and 
    !> \f$\mathbf{F}^\mathrm{x}\f$ is the TL of \f$\mathcal{F}\f$
    !> with respect to the \f$\mathbf{x}\f$ component.
    !>
    !> \b Note
    !>
    !> This should be overridden by each subclass.
    !> @param[in] self The layer.
    !> @param[in] member The index inside the batch.
    !> @param[in] dp The parameter perturbation.
    !> @param[in] dx The state perturbation.
    !> @param[out] dy The output perturbation.
    subroutine layer_apply_tangent_linear(self, member, dp, dx, dy)
        class(Layer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        print *, 'WARNING: using non-implemented method Layer::apply_tangent_linear()'
    end subroutine layer_apply_tangent_linear

    !--------------------------------------------------
    !> @brief Implements \ref layer::apply_adjoint.
    !>
    !> Applies the adjoint of the layer.
    !>
    !> @details The adjoint operator reads
    !> \f[ d\mathbf{p} = [\mathbf{F}^\mathrm{p}]^\top d\mathbf{y},\f]
    !> \f[ d\mathbf{x} = [\mathbf{F}^\mathrm{x}]^\top d\mathbf{y},\f]
    !> where \f$\mathbf{F}^\mathrm{p}\f$ is the TL of \f$\mathcal{F}\f$
    !> with respect to the \f$\mathbf{p}\f$ component and 
    !> \f$\mathbf{F}^\mathrm{x}\f$ is the TL of \f$\mathcal{F}\f$
    !> with respect to the \f$\mathbf{x}\f$ component.
    !>
    !> \b Note
    !>
    !> This should be overridden by each subclass.
    !>
    !> The intent of `dy` is declared `inout` instead of `in` because,
    !> for certain subclasses (e.g. \ref fnn_layer_dense::denselayer)
    !> the value of `dy` gets overwritten during the process (bad side-effect).
    !> @param[in] self The layer.
    !> @param[in] member The index inside the batch.
    !> @param[inout] dy The output perturbation.
    !> @param[out] dp The parameter perturbation.
    !> @param[out] dx The state perturbation.
    subroutine layer_apply_adjoint(self, member, dy, dp, dx)
        class(Layer), intent(in) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        print *, 'WARNING: using non-implemented method Layer::apply_adjoint()'
    end subroutine layer_apply_adjoint

end module fnn_layer

