import cv2

# Get user supplied values
imagePath = "images/" + "IMG_4801.JPG"

testCascPath = "haarcascades/" + "haarcascade_frontalface_default.xml"

palmCascPath = "haarcascades/" + "palm.xml"  # 5, 60,60
eyePairBigPath = "haarcascades/" + "eyePairBig.xml"  # 6, 60,60
mouthCascPath = "haarcascades/" + "mouth.xml"  # 40, 60,60
upperBodyCascPath = "haarcascades/" + "upperBody.xml"  # 20, 60,60
noseCascPath = "haarcascades/" + "nose.xml"  # 80, 60,60
fullBodyCascPath = "haarcascades/" + "fullBody.xml"  # 1, 60,60
faceCascPath = "haarcascades/" + "haarcascade_frontalface_default.xml"
fistCascPath = "haarcascades/" + "fist.xml"


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
cascs = {
    "palm": palmCascade,
    "eyePair": eyePairCascade,
    "mouth": mouthCascade,
    "upperBody": upperBodyCascade,
    "nose": noseCascade,
    "fullBody": fullBodyCascade,
    "faceCascade": faceCascade,
    "fistCascade": fistCascade,
}


feature_list = [
    "palm",
    "eyePair",
    "mouth",
    "upperBody",
    "nose",
    "fullBody",
    "face",
    "fist",
]
features_found = [0] * len(feature_list)
rects_list = [[]] * len(feature_list)

# list of minNeighbors
minNeighbors = [75, 3, 40, 20, 80, 1, 40, 75]  # 92,

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

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Check Readiness
checkReady = ["fullBody", "upperBody", "eyePairs"]
for op in checkReady:
    pass


for i in range(0, len(cascs)):
    # detect the feature in question and save all instances as rects
    rects = cascs[i].detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=minNeighbors[i], minSize=(60, 60)
    )

    # Save all the rects:
    rects_list[i] = rects

    # Draw a rectangle around the faces
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), colors[i], 2)
        # record that this object has been detected
        features_found[i] += 1

# Print out features and number of them that are found
for i in range(0, len(feature_list)):
    print(feature_list[i] + ": " + str(features_found[i]))


scale_percent = 20  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("Image", resized)
cv2.waitKey(0)
