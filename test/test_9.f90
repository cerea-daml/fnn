
program main

    use fnn_common
    use fnn_network_sequential

    implicit none
    integer(ik) :: Nx, Ny, Ne, i
    type(SequentialNeuralNetwork) :: network

    real(rk), allocatable :: x(:, :), y(:, :)

    Ne = 100
    network = snn_fromfile(Ne, 'test_9_model.txt', 'test_9_model.bin')
    Nx = network % get_input_size()
    Ny = network % get_output_size()

    allocate(x(Nx, Ne))
    allocate(y(Ny, Ne))

    call rand2d(x)
    
    do i = 1, Ne
        call network % apply_forward(.true., i, x(:, i), y(:, i))
    end do

    open(unit=10, file='test_9_out.bin', form='unformatted', access='stream', action='write')
    write(10) x
    write(10) y
    close(10)

end program main

