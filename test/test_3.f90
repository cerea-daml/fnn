
program main

    use fnn_common
    use fnn_network_sequential

    implicit none
    integer(ik) :: Nx, Ny, Ne, i
    type(SequentialNeuralNetwork) :: network

    real(rk), allocatable :: x(:, :), y(:, :)

    Ne = 100
    network = snn_fromfile(Ne, 'test_3_model.txt')
    Nx = network % get_input_size()
    Ny = network % get_output_size()

    allocate(x(Nx, Ne))
    allocate(y(Ny, Ne))

    call rand2d(x)
    
    do i = 1, Ne
        call network % apply_forward(.true., i, x(:, i), y(:, i))
    end do

    open(unit=1, file='test_3_out.bin', form='unformatted')
    write(1) x
    write(1) y
    close(1)

end program main

