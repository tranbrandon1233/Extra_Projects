func findNotificationID(_ notifications: [Int], _ targetID: Int) -> Int {
    var left = 0
    var right = notifications.count - 1
    while left <= right {
        let mid = left + (right - left) / 2
        if notifications[mid] == targetID {
            return mid
        }
        if notifications[left] <= notifications[mid] {
            if notifications[left] <= targetID && targetID < notifications[mid] {
                right = mid - 1
            } else {
                left = mid + 1
            }
        } else {
            if notifications[mid] < targetID && targetID <= notifications[right] {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
    }
    return -1
}

print(findNotificationID([4, 5, 6, 7, 0, 1, 2],1)) // 5