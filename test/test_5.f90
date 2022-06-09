
program main

    use fnn_common
    use fnn_network_sequential

    implicit none
    integer(ik) :: Nx, Ny, Ne, i, Np
    type(SequentialNeuralNetwork) :: network

    real(rk), allocatable :: x(:, :), y(:, :), dy(:, :), dp(:, :), dx(:, :)

    Ne = 100
    network = snn_fromfile(Ne, 'test_5_model.txt')
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


    open(unit=1, file='test_5_out.bin', form='unformatted')
    write(1) x
    write(1) y
    write(1) dy
    write(1) dp
    write(1) dx
    close(1)

end program main

