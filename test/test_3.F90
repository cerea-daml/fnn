
program main

    use mod_kinds, only: ik, rk
    use mod_network, only: SequentialNeuralNetwork, snn_fromfile
    use mod_random

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

    call rand(x)
    
    do i = 1, Ne
        call network % forward(i, x(:, i), y(:, i))
    end do

    open(unit=1, file='test_3_out.bin', form='unformatted')
    write(1) x
    write(1) y
    close(1)

end program main

