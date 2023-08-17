
program main

    use fnn_common
    use fnn_network_sequential

    implicit none
    integer(ik) :: Nx, Ny, Ne, i, Np
    type(SequentialNeuralNetwork) :: network

    real(rk), allocatable :: x(:, :), y(:, :), dp(:), dx(:, :), dy(:, :)

    Ne = 100
    network = snn_fromfile(Ne, 'test_4_model.txt', 'test_4_model.bin')
    Nx = network % get_input_size()
    Ny = network % get_output_size()
    Np = network % get_num_parameters()

    allocate(x(Nx, Ne))
    allocate(y(Ny, Ne))
    allocate(dp(Np))
    allocate(dx(Nx, Ne))
    allocate(dy(Ny, Ne))

    call rand2d(x)
    call rand1d(dp)
    call rand2d(dx)
    
    do i = 1, Ne
        call network % apply_forward(.true., i, x(:, i), y(:, i))
    end do

    do i = 1, Ne
        call network % apply_tangent_linear(i, dp, dx(:, i), dy(:, i))
    end do


    open(unit=10, file='test_4_out.bin', form='unformatted', access='stream', action='write')
    write(10) x
    write(10) y
    write(10) dp
    write(10) dx
    write(10) dy
    close(10)

end program main

