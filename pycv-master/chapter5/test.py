def original():

    # 这是我们写图像的地方，如果给出了output_dir
    # 在命令行中：

    out_dir = None
    names = ['Joe', 'Jane', 'Jack']
    # 您至少需要一条指向图像数据的路径，请参阅
    # 关于如何准备这个源代码的教程
    # 你的图像数据：

    if len(sys.argv) < 2:
        print("USAGE: facerec_demo.py </path/to/images> [</path/to/store/images/at>]")
        sys.exit()

    # Now read in the image data. This must be a valid path!
    [X,y] = read_images(sys.argv[1])
    # 将标签转换为32位整数。 这是64位机器的解决方法，
    # 因为标签会被截断。 这将在代码中修复为
    # 尽快，所以Python用户不需要知道这一点。
    y = np.asarray(y, dtype=np.int32)
    # If a out_dir is given, set it:
    if len(sys.argv) == 3:
        out_dir = sys.argv[2]
    # 创建Eigenfaces模型。 我们将使用默认值
    # 参数这个简单的例子，请阅读文档
    # thresholdolding：
    #model = cv2.face.createLBPHFaceRecognizer()
    model = cv2.face.createEigenFaceRecognizer()
    #  创建人脸识别模型
    #  函数返回Python列表，
    #  使用np.asarray将它们变成NumPy列表来制作
    #  eigenface 识别时  两个可以设置的重要参数
    #  要保留的主成分数目  指定的置信度阈值（浮点数）

    #  接下来 重复与人脸检测操作类似的过程
    #  通过检测到人脸进行人脸识别  从而扩展帧处理过程
    #  最后在调整后的区域调用  predict（）函数
    model.train(np.asarray(X), np.asarray(y))
    # 我们现在从模型中得到预测！ 实际上你
    # 应始终使用看不见的图像来测试您的模型。
    # 但当我切片时，很多人都感到困惑

    # 在C + +版本中关闭，所以我只使用我们的图像
    # 接受过训练。
    # model.predict 将返回预测的标签和置信度
    # predict() 函数返回含有两个元素的数组  第一个元素是所有识别个体的标签
    # 第二个是置信度评分。所有的算法都有一个置信度评分阈值，置信度评分用来衡量所识别人脸与
    # 原模型的差距  0表示完全匹配
    # 可能有时候不想保留所有的识别结果 则进一步处理，因此可用自己的算法来估算识别的置信度评分
    # 如，如果正在试图识别视频中的人，则可能要分析后续帧的置信度评分来估计识别是否成功
    # 在这种情况下 可以通过检查得到的置信度评分，然后得出自己的结论
    # Eigenface/Firstfaces 和LBPH的置信度评分不同
    # 前者产生0~20000的值，任意低于4000~5000的评分都是相当可靠的识别
    # LBPH有类似的工作方式，但是一个好的识别参考值要低于50，任何高于80的参考值都是较低的置信度评分
    #

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    while (True):
      read, img = camera.read()
      faces = face_cascade.detectMultiScale(img, 1.3, 5)
      for (x, y, w, h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = gray[x:x+w, y:y+h]
        roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
        print(roi.shape)
        params = model.predict(roi)
        print("Label: %s, Confidence: %.2f" % (params[0], params[1]))
        cv2.putText(img, names[params[0]], (x,y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
      cv2.imshow("camera", img)
      if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
        break

    [p_label, p_confidence] = model.predict(np.asarray(X[0]))
    # Print it:
    print("Predicted label = %d (confidence=%.2f)" % (p_label, p_confidence))

    # 最后我们将绘制特征脸，因为那是
    # 大多数人在报纸上看到的内容都很敏感。
    # 就像在C + +中一样，您可以访问所有模型内部
    # 数据，因为cv:: FaceRecognizer是一个cv:: Algorithm。
    # 您可以使用getParams（）查看可用参数：

    print(model.getParams())
    # Now let's get some data:
    mean = model.getMat("mean")
    eigenvectors = model.getMat("eigenvectors")
    # 我们将通过首先将其标准化来保存均值：
    mean_norm = normalize(mean, 0, 255, dtype=np.uint8)
    mean_resized = mean_norm.reshape(X[0].shape)
    if out_dir is None:
        cv2.imshow("mean", mean_resized)
    else:
        cv2.imwrite("%s/mean.png" % (out_dir), mean_resized)
    # 将第一个（最多）16
    # 个特征向量转换为灰度
    #      图片。 你也可以在这里使用cv:: normalize，但坚持下去
    #      NumPy现在要容易得多。
    #      注意：特征向量按列存储：
    for i in xrange(min(len(X), 16)):
        eigenvector_i = eigenvectors[:,i].reshape(X[0].shape)
        eigenvector_i_norm = normalize(eigenvector_i, 0, 255, dtype=np.uint8)
        # Show or save the images:
        if out_dir is None:
            cv2.imshow("%s/eigenface_%d" % (out_dir,i), eigenvector_i_norm)
        else:
            cv2.imwrite("%s/eigenface_%d.png" % (out_dir,i), eigenvector_i_norm)
    # Show the images:
    if out_dir is None:
        cv2.waitKey(0)
    cv2.destroyAllWindows()