
program main

    use mod_kinds, only: ik, rk
    use mod_network, only: SequentialNeuralNetwork, snn_fromfile
    use mod_random

    implicit none
    integer(ik) :: Nx, Ny, Ne, i, Np
    type(SequentialNeuralNetwork) :: network

    real(rk), allocatable :: x(:, :), y(:, :), new_p(:)

    Ne = 100
    network = snn_fromfile(Ne, 'test_6_model.txt')
    Nx = network % get_input_size()
    Ny = network % get_output_size()
    Np = network % get_num_parameters()

    allocate(x(Nx, Ne))
    allocate(y(Ny, Ne))
    allocate(new_p(Np))

    call rand(x)
    call rand(new_p)
    call network % set_parameters(new_p)
    
    do i = 1, Ne
        call network % forward(i, x(:, i), y(:, i))
    end do

    open(unit=1, file='test_6_out.bin', form='unformatted')
    write(1) new_p
    write(1) x
    write(1) y
    close(1)

end program main

