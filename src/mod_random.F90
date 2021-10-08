
module mod_random

    use mod_kinds, only: ik, rk

    implicit none

    private
    public :: rand

    interface rand
        ! interface to 1d and 2d rand subroutines
        module procedure :: rand1d, rand2d
    end interface rand

contains

    subroutine rand1d(x)
        ! fill a 1d table with random numbers
        real(rk), intent(out) :: x(:)
        call random_number(x)
    end subroutine rand1d

    subroutine rand2d(x)
        ! fill a 2d table with random numbers
        real(rk), intent(out) :: x(:, :)
        call random_number(x)
    end subroutine rand2d

end module mod_random

