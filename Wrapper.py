import cv2
import glob
import numpy as np
import scipy.optimize as opt
import scipy

def calcHomography(images, world_pts, save_folder):
    copy=np.copy(images)
    allH = []
    img_pts = []
    for i, img in enumerate(copy):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret: 
            corners = corners.reshape(-1, 2)
            H, _ = cv2.findHomography(world_pts, corners, cv2.RANSAC, 5.0) #finds homography between given world points/
            allH.append(H)
            img_pts.append(corners)
            cv2.drawChessboardCorners(img, (9, 6), corners, True)
            img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
            cv2.imwrite(save_folder + '/' + str(i) + '_corners.png', img)
    return allH, img_pts

def v(m, n, H):
    return np.array([
        H[0][m] * H[0][n],
        H[0][m] * H[1][n] + H[1][m] * H[0][n],
        H[1][m] * H[1][n],
        H[2][m] * H[0][n] + H[0][m] * H[2][n],
        H[2][m] * H[1][n] + H[1][m] * H[2][n],
        H[2,][m] * H[2][n]
    ])

def getIntrinsic(allH):
    V = []
    for h in allH:
        V.append(v(0, 1, h))
        V.append(v(0, 0, h) - v(1, 1, h))
    V = np.array(V)
    # print(V)
    _, _, vt = np.linalg.svd(V)

    b = vt[-1][:]
    # print(b)
    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]

    v0 = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
    # print(v0)
    lamda_i = B33 - (B13**2 + v0*(B12*B13 - B11*B23))/B11
    # print(lamda_i)
    alpha = np.sqrt(lamda_i/B11)
    # print(alpha)
    beta = np.sqrt(lamda_i*B11 / (B11*B22 - B12**2))
    # print(beta)
    gamma = -1*B12*(alpha**2)*beta/lamda_i
    # print(gamma)
    u0 = (gamma*v0/beta) - (B13*(alpha**2)/lamda_i)

    A = np.array([[alpha, gamma, u0],
                      [0, beta, v0],
                      [0, 0, 1]])
    return A


def getExtrinsic(A, allH):
    A_inv = np.linalg.inv(A)
    extrinsics = []
    for H in allH:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        lamda_e = 1/scipy.linalg.norm((A_inv @ h1), 2)
        # print(lamda_e)
        r1 = lamda_e * (A_inv @ h1) 
        # print(r1)
        r2 = lamda_e * (A_inv @ h2) 
        # print(r2)
        # r3 = np.cross(r1, r2)
        # print(r3)
        t = lamda_e * (A_inv @ h3)
        Rt = np.vstack((r1, r2, t)).T
        # print(R)
        extrinsics.append(Rt)
    return extrinsics


def loss(A, img_pts, world_pts, extrinsics):
    alpha, gamma, beta, u0, v0, k1, k2 = A
    A = np.array([[alpha, gamma, u0],
                      [0, beta, v0],
                      [k1, k2, 1]])
    final_error = []

    for i, img_pt in enumerate(img_pts):

        A_Rt = (A @ extrinsics[i].reshape(3,3))
          
        serror = 0
        for j in range(len(world_pts)):
            world_pt = world_pts[j]

            M = np.array([world_pt[0], world_pt[1], 1]).reshape(3, 1)

            proj_pts = (extrinsics[i] @ M) #Equation 1
            proj_pts /= proj_pts[2]
            x, y = proj_pts[0], proj_pts[1]

            N = (A_Rt @ M)
            u, v = N[0]/N[2], N[1]/N[2]
            # print("u",u)

            mij = img_pt[j]
            mij = np.array([mij[0], mij[1], 1], dtype = np.float32)

            t = x**2 + y**2
            # print(x)
            u_cap = u + (u-u0)*(k1*t + k2*(t**2))
            # print("u_cap", u_cap)
            v_cap = v + (v-v0)*(k1*t + k2*(t**2))
            mij_cap = np.array([u_cap, v_cap, 1], dtype = np.float32)

            error = scipy.linalg.norm((mij- mij_cap), 2)
            serror += error
        final_error.append(serror / 54)
        # print(final_error)
    return np.array(final_error)

def optimization(A, img_pts, world_pts, extrinsics):
    alpha = A[0, 0]
    gamma = A[0, 1]
    u0 = A[0, 2]
    beta = A[1, 1]
    v0 = A[1, 2]
    k1 = A[2, 0]
    k2 = A[2, 1]
    optimized = scipy.optimize.least_squares(fun=loss, x0 = [alpha, gamma, beta, u0, v0, k1, k2], method = 'lm', args=(img_pts, world_pts, extrinsics))
    [alpha_opt, gamma_opt, beta_opt, u0_opt, v0_opt, k1_opt, k2_opt] = optimized.x
    print("k1 opt", k1_opt)
    print("k2 opt", k2_opt)
    A_opt = np.array([[alpha_opt, gamma_opt, u0_opt],
                      [0, beta_opt, v0_opt],
                      [0, 0, 1]])
    return A_opt, k1_opt, k2_opt

def error(A_opt, distortion, extrinsic, img_pts, world_pts):
    u0 = A_opt[0][2]
    v0 = A_opt[1][2]
    k1, k2= distortion[0], distortion[1]

    re_error = []
    reproj_pts = []
    for i, img_pt in enumerate(img_pts):
        A_Rt = A @ extrinsic[i]

        error_sum = 0
        reproj_pts = []
        for j in range(world_pts.shape[0]):
            world_pt = world_pts[j]
            M = np.array([world_pt[0], world_pt[1], 1]).reshape(3, 1)

            proj_pts = (extrinsics[i] @ M)
            x, y = proj_pts[0]/proj_pts[2], proj_pts[1]/proj_pts[2]

            N = (A_Rt @ M)
            u, v = N[0]/N[2], N[1]/N[2]

            mij = img_pt[j]
            mij = np.array([mij[0], mij[1], 1])

            t = x**2 + y**2
            u_cap = u + (u-u0)*(k1*t + k2*(t**2))
            v_cap = v + (v-v0)*(k1*t + k2*(t**2))

            reproj_pts.append([int(u_cap), int(v_cap)])
            mij_cap = np.array([u_cap, v_cap, 1])
            error = scipy.linalg.norm((mij - mij_cap), ord=2)
            print("error", error)
            error_sum += error

        re_error.append(error_sum)
        reproj_pts.append(reproj_pts)

    re_error_avg = np.sum(np.array(re_error)) / (len(img_pts) * world_pts.shape[0])
    return re_error_avg, reproj_pts

if __name__ == '__main__':
    Data = 'Calibration_Imgs/'
    Save = 'Ouputs/'

    size = 21.5  #size of chessboard image given

    images = [cv2.imread(file) for file in glob.glob(Data + '*.jpg')] #given checkerboard images

    world_pts_x, world_pts_y = np.meshgrid(range(9), range(6)) #given checkerboard pattern
    world_pts = np.array(np.hstack((world_pts_x.reshape(54, 1), world_pts_y.reshape(54, 1))).astype(np.float32)*size)

    allH, img_pts = calcHomography(images, world_pts, Save) #homography calculation 

    A = getIntrinsic(allH)
    print("K matrix: \n", A)
    extrinsics= getExtrinsic(A, allH)

    A_opt, k1, k2 = optimization(A, img_pts, world_pts, extrinsics)
    distortion_opt = np.array([k1, k2]).reshape(2, 1)
    extrinsic_opt = getExtrinsic(A_opt, allH)

    reprojected_error, reprojected_pts = error(A_opt, distortion_opt, extrinsic_opt, img_pts, world_pts)
    
    K = np.array(A_opt, np.float32).reshape(3,3)
    distortion = np.array([distortion_opt[0],distortion_opt[1], 0, 0], np.float32)

    print("K matrix: \n", K)
    print('Distortion optimized: ', distortion_opt)
    print("Error: ", reprojected_error)

    for i, image_points in enumerate(reprojected_pts):
        image = cv2.undistort(images[i], K, distortion)
        for point in image_points:
            image = cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), 10)

        cv2.imwrite(Save + "rectified_" + str(i) + ".png", image)

