
program main

    use fnn_common
    use fnn_network_sequential

    implicit none
    integer(ik) :: Nx, Ny, Ne, i, Np
    type(SequentialNeuralNetwork) :: network

    real(rk), allocatable :: x(:, :), y(:, :), dy(:, :), dp(:, :), dx(:, :)

    Ne = 100
    network = snn_fromfile(Ne, 'test_5_model.txt', 'test_5_model.bin')
    Nx = network % get_input_size()
    Ny = network % get_output_size()
    Np = network % get_num_parameters()

    allocate(x(Nx, Ne))
    allocate(y(Ny, Ne))
    allocate(dp(Np, Ne))
    allocate(dx(Nx, Ne))
    allocate(dy(Ny, Ne))

    call rand2d(x)
    call rand2d(dy)
    
    do i = 1, Ne
        call network % apply_forward(.true., i, x(:, i), y(:, i))
    end do

    do i = 1, Ne
        call network % apply_adjoint(i, dy(:, i), dp(:, i), dx(:, i))
    end do


    open(unit=10, file='test_5_out.bin', form='unformatted', access='stream', action='write')
    write(10) x
    write(10) y
    write(10) dy
    write(10) dp
    write(10) dx
    close(10)

end program main

