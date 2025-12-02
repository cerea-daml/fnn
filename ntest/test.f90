
program main

    use fnn

    implicit none
    integer(ik) :: Nx, Ny, Ne, i
    type(NeuralNetwork) :: network

    real(rk), allocatable :: x(:, :), y(:, :)

    Ne = 100
    network = nn_fromfile(Ne, 'test_3_model.txt', 'test_3_model.bin')
    Nx = network % get_input_size()
    Ny = network % get_output_size()

    allocate(x(Nx, Ne))
    allocate(y(Ny, Ne))

    call rand2d(x)
    
    do i = 1, Ne
        call network % apply_forward(.true., i, x(:, i), y(:, i))
    end do

    open(unit=10, file='test_3_out.bin', form='unformatted', access='stream', action='write')
    write(10) x
    write(10) y
    close(10)

end program main

