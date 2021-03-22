import cirq
from period_finding import period_finding_routine
from period_finding import euclid_gcd
import numpy as np
print('numpy version',np.__version__)


class Factoring:
    """"
    Find the factorization of number N = p*q
    where p and q are prime to each other  
    """

    def __init__(self, N):
        self.N = N

    def factoring(self):

        prev_trials_for_a = []
        factored = False

        while not factored:
            new_a_found = False

            # Sample a new "a" not already sampled 
            while not new_a_found:
                a = np.random.randint(2, self.N)
                if a not in prev_trials_for_a:
                    new_a_found = True

            # "a" not co-prime to N are not periodic 
            if euclid_gcd(self.N, a) == 1:
                # Call the period_finding_routine from PeriodFinding 
                # Implementation
                period = period_finding_routine(a=a, N=self.N)

                # Check if the period is even.
                # It period even (a^(r/2))^2 = 1 mod (N) 
                # for integer a^(r/2)
                if period % 2 == 0:

                    # Check if a^(r/2) != +/- 1 mod(N)
                    # if condition satisfied number gets 
                    # factorized in this iteration
                    if a ** (period / 2) % self.N not in [+1, -1]:
                        prime_1 = euclid_gcd(self.N, a ** (period / 2) + 1)
                        prime_2 = euclid_gcd(self.N, a ** (period / 2) - 1)
                        factored = True
                        return prime_1, prime_2
            else:
                # If we have exhausted all "a"s and
                # still havent got prime factors something 
                # is off
                if len(prev_trials_for_a) == self.N - 2:
                    raise ValueError(f"Check input is a product of two primes")


if __name__ == '__main__':

    fp = Factoring(N=15)
    factor_1, factor_2 = fp.factoring()

    if factor_1 is not None:
        print(f"The factors of {fp.N} are {factor_1} and {factor_2}")
    else:
        print(f"Error in factoring")
