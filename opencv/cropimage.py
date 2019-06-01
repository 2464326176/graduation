
"""
输入：图片路径(path+filename)，裁剪获得小图片的列数、行数（也即宽、高）
输出：无
"""
# def crop_one_picture(path, filename, cols, rows):
#     img = cv2.imread(path + filename, -1)  ##读取彩色图像，图像的透明度(alpha通道)被忽略，默认参数;灰度图像;读取原始图像，包括alpha通道;可以用1，0，-1来表示
#     sum_rows = img.shape[0]  # 高度
#     sum_cols = img.shape[1]  # 宽度
#     save_path = path + "\\crop{0}_{1}\\".format(cols, rows)  # 保存的路径
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     print("裁剪所得{0}列图片，{1}行图片.".format(int(sum_cols / cols), int(sum_rows / rows)))
#
#     for i in range(int(sum_cols / cols)):
#         for j in range(int(sum_rows / rows)):
#             cv2.imwrite(
#                 save_path + os.path.splitext(filename)[0] + '_' + str(j) + '_' + str(i) + os.path.splitext(filename)[1],
#                 img[j * rows:(j + 1) * rows, i * cols:(i + 1) * cols, :])
#             # print(path+"\crop\\"+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+os.path.splitext(filename)[1])
#     print("裁剪完成，得到{0}张图片.".format(int(sum_cols / cols) * int(sum_rows / rows)))
#     print("文件保存在{0}".format(save_path))

