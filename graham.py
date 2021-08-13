import matplotlib.pyplot as plt
import math
import time
import numpy as np
import sys

start_t = time.clock()

def directionOfPoint(A, B, P):
     
     
    # Subtracting co-ordinates of
    # point A from B and P, to
    # make A as origin
    B[0] -= A[0]
    B[1] -= A[1]
    P[0] -= A[0]
    P[1] -= A[1]
  
    # Determining cross Product
    cross_product = B[0] * P[1] - B[1] * P[0]
  
    # Return RIGHT if cross product is positive
    if (cross_product > 0):
        return 1
         
    # Return LEFT if cross product is negative
    else:
    	return 0

# points the index of the smallest ordinate point, if there are more than one, return the smallest abscissa point
def get_bottom_point(points):
    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points[i][1] < points[min_index][1] or (
                points[i][1] == points[min_index][1] and points[i][0] < points[min_index][0]):
            min_index = i
    return min_index


# Sort according to the polar angle with the center point, cosine, center_point: center point
def sort_polar_angle_cos(points, center_point):
    n = len(points)
    cos_value = []
    rank = []
    norm_list = []
    for i in range(0, n):
        point_ = points[i]
        point = [point_[0] - center_point[0], point_[1] - center_point[1]]
        rank.append(i)
        norm_value = math.sqrt(point[0] * point[0] + point[1] * point[1])
        norm_list.append(norm_value)
        if norm_value == 0:
            cos_value.append(1)
        else:
            cos_value.append(point[0] / norm_value)

    for i in range(0, n - 1):
        index = i + 1
        while index > 0:
            if cos_value[index] > cos_value[index - 1] or (
                    cos_value[index] == cos_value[index - 1]
                    and norm_list[index] > norm_list[index - 1]):
                temp = cos_value[index]
                temp_rank = rank[index]
                temp_norm = norm_list[index]
                cos_value[index] = cos_value[index - 1]
                rank[index] = rank[index - 1]
                norm_list[index] = norm_list[index - 1]
                cos_value[index - 1] = temp
                rank[index - 1] = temp_rank
                norm_list[index - 1] = temp_norm
                index = index - 1
            else:
                break
    sorted_points = []
    for i in rank:
        sorted_points.append(points[i])

    return sorted_points


# Return the angle between the vector and the vector [1, 0]-how many degrees to rotate counterclockwise from [1, 0] to reach this vector
def vector_angle(vector):
    norm_ = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
    if norm_ == 0:
        return 0

    angle = math.acos(vector[0] / norm_)
    if vector[1] >= 0:
        return angle
    else:
        return 2 * math.pi - angle


# Cross product of two vectors
def coss_multi(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def graham_scan(points):
    bottom_index = get_bottom_point(points)
    bottom_point = points.pop(bottom_index)
    sorted_points = sort_polar_angle_cos(points, bottom_point)

    m = len(sorted_points)
    if m < 2:
        print("The number of points is too small to form a convex hull")
        return

    stack = []
    stack.append(bottom_point)
    stack.append(sorted_points[0])
    stack.append(sorted_points[1])
    # print('current stack', stack)

    for i in range(2, m):
        length = len(stack)
        top = stack[length - 1]
        next_top = stack[length - 2]
        v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
        v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        while coss_multi(v1, v2) >= 0:
            if length < 3:     # After adding these two lines of code, no error will be reported when the amount of data is large
                break          # After adding these two lines of code, no errors will be reported when the amount of data is large
            stack.pop()
            length = len(stack)
            top = stack[length - 1]
            next_top = stack[length - 2]
            v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
            v2 = [top[0] - next_top[0], top[1] - next_top[1]]
        stack.append(sorted_points[i])

    return stack


# Generate random points
n_iter = [5000, 10000, 15000, 20000, 25000]
time_cost = []
for n in n_iter:
    points = []
    for i in range(n):
        point_x = np.random.randint(1, 100)
        point_y = np.random.randint(1, 100)
        temp = np.hstack((point_x, point_y))
        point = temp.tolist()
        points.append(point)

    #### IMPROVISATION ####
    left = points[0]
    right = points[0]
    top = points[0]
    bottom = points[0]


    for j in range(0, n):
        if(points[j][0] < left[0]):
            left = points[j]
        if(points[j][0] > right[0]):
            right = points[j]
        if(points[j][1] > top[1]):
            top = points[j]
        if(points[j][1] < bottom[1]):
            bottom = points[j]
    
    newpoints = []

    for j in range(0, n):
        if(directionOfPoint(top, left, points[i]) == 0):
            newpoints.append(points[i])
        
        elif(directionOfPoint(right, top, points[i]) == 0):
            newpoints.append(points[i])
        
        elif(directionOfPoint(left, bottom, points[i]) == 0):
            newpoints.append(points[i])
        
        elif(directionOfPoint(bottom, right, points[i]) == 0):
            newpoints.append(points[i])
        
    

    result = graham_scan(newpoints)

    # Record program running time
    end_t = time.clock()
    time_iter = end_t - start_t
    #print("Graham-Scan algorithm running time:", time_iter)
    # draw(list_points, border_line)
    time_cost.append(time_iter)
    # Draw the result graph
    '''
    for point in points:
        plt.scatter(point[0], point[1], marker='o', c='y', s=8)
    length = len(result)
    for i in range(0, length - 1):
        plt.plot([result[i][0], result[i + 1][0]], [result[i][1], result[i + 1][1]], c='r')
    plt.plot([result[0][0], result[length - 1][0]], [result[0][1], result[length - 1][1]], c='r')
    plt.show()
    '''

# Running time under different test sets
plt.plot(n_iter, time_cost)
plt.show()
