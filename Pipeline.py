import cv2

# Get user supplied values
imgs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

imageFolder = "collected-images/newImages/images/"
testNeb = 20
mSize = (85, 85)
testCascPath = "haarcascades/" + "fist.xml"

palmCascPath = "haarcascades/" + "palm.xml"  # 20, 85,85 (DONE)
eyePairBigPath = "haarcascades/" + "eyePairBig.xml"  # 20, 30,30 (DONE)
mouthCascPath = "haarcascades/" + "mouth.xml"  ###80, 60,60 (DONE)
upperBodyCascPath = "haarcascades/" + "upperBody.xml"  # 50, 500,500 (DONE)
noseCascPath = "haarcascades/" + "nose.xml"  ###80, 60,60 (DONE)
fullBodyCascPath = "haarcascades/" + "fullBody.xml"  ###1, 60,60 (DONE)
faceCascPath = (
    "haarcascades/" + "haarcascade_frontalface_default.xml"
)  # 25, 60,60 (DONE)
fistCascPath = "haarcascades/" + "fist.xml"  # 20 85,85 (DONE)

minSize = [
    (85, 85),
    (30, 30),
    (60, 60),
    (500, 500),
    (60, 60),
    (60, 60),
    (60, 60),
    (85, 85),
]

# list of minNeighbors
minNeighbors = [20, 20, 80, 50, 80, 1, 25, 20]  # 92,

# Create the haar cascades
testCascade = cv2.CascadeClassifier(testCascPath)

palmCascade = cv2.CascadeClassifier(palmCascPath)
eyePairCascade = cv2.CascadeClassifier(eyePairBigPath)
mouthCascade = cv2.CascadeClassifier(mouthCascPath)
upperBodyCascade = cv2.CascadeClassifier(upperBodyCascPath)
noseCascade = cv2.CascadeClassifier(noseCascPath)
fullBodyCascade = cv2.CascadeClassifier(fullBodyCascPath)
faceCascade = cv2.CascadeClassifier(faceCascPath)
fistCascade = cv2.CascadeClassifier(fistCascPath)

cascs = [
    palmCascade,
    eyePairCascade,
    mouthCascade,
    upperBodyCascade,
    noseCascade,
    fullBodyCascade,
    faceCascade,
    fistCascade,
]


# list of colors
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
    (120, 50, 200),
]

for img in imgs:

    imagePath = imageFolder + str(img) + ".jpg"

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    """
    rects = testCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=testNeb,
            minSize=mSize
        )
    # Draw a rectangle around the faces
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 2)

    """
    for i in range(0, len(cascs)):
        rects = cascs[i].detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=minNeighbors[i], minSize=minSize[i]
        )
        # Draw a rectangle around the faces
        for (x, y, w, h) in rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), colors[i], 2)

    """
    scale_percent = 20 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Image"+str(img), resized)
    """
    folder = "outputs"
    cv2.imwrite(folder + "/" + "out" + str(img) + ".jpg", image)

    # cv2.waitKey(0)

# cv2.waitKey(0)
