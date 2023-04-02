import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

# the starting function for questions 1 and 2
def function(t: float, y: float):
    return t - (y**2)

# Eulers Method functions and opperations
def do_work_euler(t, y, h):
    basic_function_call = function(t, y)
    incremented_y = y + (h * basic_function_call)
    incremented_t = t + h
    return incremented_t, incremented_y

# Eulers Method
def euler_method():
    original_y = 1
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10

    # set up h
    h = (end_of_t - start_of_t) / num_of_iterations

    # make t and their starting values
    t = start_of_t
    y = original_y

    for cur_iteration in range(0, num_of_iterations):
        # this is to calculate the next value of y and t
        t, y = do_work_euler(t, y, h)

    return y

# Question 2
# runge_kutta functions and opperations
def do_work_runge_kutta(t, y, h):
    basic_function_call = function(t, y)
    k1 = basic_function_call
    k2 = function(t + h/2, y + h/2 * k1)
    k3 = function(t + h/2, y + h/2 * k2)
    k4 = function(t + h, y + h * k3)
    incremented_y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    incremented_t = t + h
    return incremented_t, incremented_y

# Runge-Kutta does that same as Eulers Method for this part
def runge_kutta():
    original_y = 1
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10

    # set up h
    h = (end_of_t - start_of_t) / num_of_iterations

    # initialize t and y to their starting values
    t = start_of_t
    y = original_y

    for cur_iteration in range(0, num_of_iterations):
        # calculate the next value of y and t using the modified do_work function
        t, y = do_work_runge_kutta(t, y, h)

    return y

# Question 3
# Gaussian elimination method
def gaussian_elimination(A, b):
    n = len(b)
    A = A.astype(np.float64)
    b = b.astype(np.float64)
    # Combine A and b into augmented matrix
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)
    # Perform elimination
    for i in range(n):
        # Eliminate entries below pivot
        for j in range(i+1, n):
            # Calculate the factor for row j and column i
            # the pivot part of the matrix
            pivot = Ab[i, i]
            # the parts of the matrix that need to be zeroed out
            entry_to_zero_out = Ab[j, i]
            # the factor
            # multiply this to row i and subtract from j
            # this is the factor = Ab[j,i] / Ab[i,i]
            factor = entry_to_zero_out / pivot
            Ab[j,:] -= factor * Ab[i,:] # operation 2 of row operations from notes
    # Backward substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        # Setting up dot product
        # the coefficients of what's remaining
        coefficients = Ab[i,i+1:n]
        # the variables of what's remaining
        variables = x[i+1:n]
        # the dot product
        dot_product = np.dot(coefficients, variables)

        # Finding the i'th value
        # right hand side of the i'th value
        rhs_i = Ab[i, n]
        # numerator for i-th
        numerator = rhs_i - dot_product
        # denominator for i-th
        denominator = Ab[i,i]
        # division
        ith_value = numerator / denominator
        # converting it to an integer
        ith_int_value = int(ith_value)
        # i'th varible
        x[i] = ith_int_value

    return x


# Question 4
# LU factorization with forward and backward substitution

def lu_factorization(A_matrix):
    # Size of matrix A
    matrix_size = len(A_matrix)
    
    # Defining the Lower and Upper Matrix
    lower_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)
    upper_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)

    # LU factorization
    for i in range(matrix_size):
        # Need to set diagonal elements of L to 1
        lower_matrix[i, i] = 1
        for j in range(i, matrix_size):
            # Dot prodct for upper matrix
            dot_product_for_upper_matrix = sum(lower_matrix[i, k] * upper_matrix[k, j] for k in range(i))
            # This is the jth element of the ith row of U
            upper_matrix[i, j] = A_matrix[i, j] - dot_product_for_upper_matrix

        for j in range(i+1, matrix_size):
            # Dot prodct for lower matrix
            dot_product_for_lower_matrix = sum(lower_matrix[j, k] * upper_matrix[k, i] for k in range(i))
            # This is the ith element of the jth row of L
            lower_matrix[j, i] = (A_matrix[j, i] - dot_product_for_lower_matrix) / upper_matrix[i, i]

    return lower_matrix, upper_matrix

# Question 5
# diagonally dominant function
def is_diagonally_dominant(matrix):
    for i in range(len(matrix)):
        # Absolute value of the i'th diagonal
        pivot_value = abs(matrix[i][i])
        
        # The sum of the absolute values of the row excluding the pivot value 
        other_values_sum = sum(abs(matrix[i][j]) for j in range(len(matrix)) if j != i)
        
        # If the pivot is less than the sum of everything else in the row
        # then the matrix is not diagonally dominant
        if pivot_value < other_values_sum:
            return False
    return True

# partial pivoting if matrix needs it
# in this case it doesn't but doesnt hurt to include it for reference later
def partial_pivot(matrix):
    # number of rows in the matrix
    num_rows = len(matrix)
    
    # goes through each row of the matrix
    for i in range(num_rows):
        # pivot_row equals current row index
        pivot_row = i
        
        # This finds the largest value in the row below the diagonal
        for j in range(i+1, num_rows):
            # checks if pivot is largest value
            if abs(matrix[j][i]) > abs(matrix[pivot_row][i]):
                pivot_row = j
        # swaps rows if pivot is not the largest value
        if pivot_row != i:
            matrix[[i, pivot_row]] = matrix[[pivot_row, i]]

    return matrix


# Question 6
# positive definite function
def is_positive_definite(A):
        A = np.array(A)

        # Checking if the matrix is square
        if len(A) != len(A[0]):
            #print("False: The matrix is not square")
            return False
        
        # Checking if the matrix is symmetric
        if not np.array_equal(A, np.transpose(A)):
            #print("False: The matrix is not symmetric")
            return False
        
        # checking the determinant doesn't equal 0
        if np.linalg.det(A) == 0:
            #print("False: determinant = 0")
            return False
        
        # Checking if each cell's absolute value less is than or equal to
        # its row's pivot value
        for i in range(len(A)):
            pivot_value = abs(A[i][i])
            if not all(abs(A[i][j]) <= pivot_value for j in range(len(A)) if j != i):
                #print("False: A value is greater than the pivot value")
                return False

        # Checking that all eigenvalues are positive meaning we may not need the
        # two checks above becasue this should meet those conditions.
        if any(np.linalg.eigvals(A) <= 0):
            #print("False: Eigenvalues <= 0")
            return False
    
        return True


if __name__ == "__main__":
    # Question 1 Answer:
    print("%.5f\n" % euler_method())

    # Question 2 Answer: 
    print("%.5f\n" % runge_kutta())


    # Question 3 Answer:
    A = np.array([[2, -1, 1],
                  [1, 3, 1],
                  [-1, 5, 4]], dtype=np.double)
    b = np.array([6, 0, -3], dtype=np.double)
    x = gaussian_elimination(A, b)
    print(str(x) + '\n')


    # Question 4 Answer:
    A_matrix = np.array([[1, 1, 0, 3],
                         [2, 1, -1, 1],
                         [3, -1, -1, 2],
                         [-1, 2, 3, -1]], dtype=np.float64)

    # LU factorization
    L, U = lu_factorization(A_matrix)
    # matrix determinant
    determinant = np.prod(np.diagonal(U))
    print("%.5f\n" % determinant)
    # L matrix
    print(str(L) + '\n')
    # U matrix
    print(str(U) + '\n')


    # Question 5 Answer
    matrix = np.array([[9, 0, 5, 2, 1],
                    [3, 9, 1, 2, 1],
                    [0, 1, 7, 2, 3],
                    [4, 2, 3, 12, 2],
                    [3, 2, 4, 0, 8]])

    # Apply partial pivoting
    pivoted_matrix = partial_pivot(matrix)

    # Check if the pivoted matrix is diagonally dominant
    if is_diagonally_dominant(pivoted_matrix):
        print("True\n")
    else:
        print("False\n")
    # The anser is false becasue of the last row
    # pivot 8 < sum of abs numbers = row sum of 9

    # Question 6 Answer:
    A = [[2, 2, 1], 
         [2, 3, 0], 
         [1, 0, 2]]

    # True means Positive Definite
    if is_positive_definite(A):
        print("True\n")
    else:
        # False means it is NOT Positive Definite 
        print("False\n")
