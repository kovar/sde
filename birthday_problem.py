from math import factorial
import matplotlib.pyplot as plt

def count_prob_of_no_common_birthdays(n):
    return factorial(365) / (factorial(365 - n) * 365 ** n)


if __name__ == "__main__":
    probabilities = []
    for n in range(1, 365 + 1):
        print(f"Probability of {n} people not having the same birthday is:", count_prob_of_no_common_birthdays(n))
        probabilities.append(count_prob_of_no_common_birthdays(n))

# plt.plot(probabilities)
# plt.show()
