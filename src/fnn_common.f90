
!> @brief Module `fnn_common`. Implements some common features.
!> @details We should implement a common interface to rand1d and
!> rand2d, but doxygen does not handle Fortran interfaces...
module fnn_common

    use iso_fortran_env, only: int32, int64, real32, real64, real128

    implicit none

    private
    public :: rk, ik, rand1d, rand2d

    !> The precision for real numbers.
    integer, parameter :: rk = real64

    !> The precision for integer numbers.
    integer, parameter :: ik = int32

contains

    !> @brief Fills a 1d table with random numbers.
    !> @details The random numbers are drawn from a uniform 
    !> distribution in [0, 1].
    !> @param[out] x The 1d table of real numbers to fill.
    subroutine rand1d(x)
        real(rk), intent(out) :: x(:)
        call random_number(x)
    end subroutine rand1d

    !> @brief Fills a 2d table with random numbers.
    !> @details The random numbers are drawn from a uniform 
    !> distribution in [0, 1].
    !> @param[out] x The 2d table of real numbers to fill.
    subroutine rand2d(x)
        real(rk), intent(out) :: x(:, :)
        call random_number(x)
    end subroutine rand2d

end module fnn_common

