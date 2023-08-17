
program main

    use fnn_common
    use fnn_network_sequential

    implicit none
    integer(ik) :: Nx, Ny, Ne, i, Np
    type(SequentialNeuralNetwork) :: network

    real(rk), allocatable :: x(:, :), y(:, :), new_p(:)

    Ne = 100
    network = snn_fromfile(Ne, 'test_6_model.txt', 'test_6_model.bin')
    Nx = network % get_input_size()
    Ny = network % get_output_size()
    Np = network % get_num_parameters()

    allocate(x(Nx, Ne))
    allocate(y(Ny, Ne))
    allocate(new_p(Np))

    call rand2d(x)
    call rand1d(new_p)
    call network % set_parameters(new_p)
    
    do i = 1, Ne
        call network % apply_forward(.true., i, x(:, i), y(:, i))
    end do

    open(unit=10, file='test_6_out.bin', form='unformatted', access='stream', action='write')
    write(10) new_p
    write(10) x
    write(10) y
    close(10)

end program main

