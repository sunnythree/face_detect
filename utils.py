import sys
import time
import os

def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    if l1 > l2:
        left = l1
    else:
        left = l2
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    if r1 < r2:
        right = r1
    else:
        right = r2
    return right - left

def box_intersection(a, b):
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    if w < 0 or h < 0:
        return 0
    area = w * h
    return area


def box_union(a, b):
    i = box_intersection(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return u


def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def confidence_box(a):
    return a[0]

def nms(bboxes, confidence, iou_thresh):
    nms_boxes = []
    bboxes = sorted(bboxes, key=confidence_box, reverse=True)
    for box in bboxes:
        if box[0] < confidence:
            continue
        is_contain = False
        for i in range(len(nms_boxes)):
            nms_box = nms_boxes[i]
            iou = box_iou(box, nms_box)
            if iou > iou_thresh:
                is_contain = True
        if not is_contain:
            nms_boxes.append(box)
    return nms_boxes


term_width = 100
try:
    _, term_width = os.popen('stty size', 'r').read().split()
except:
    print("get stty size error")
    term_width = 100
term_width = int(term_width)
if term_width <= 0:
    term_width = 100

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    sys.stdout.write(' %d/%d ' % (current + 1, total))

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))

    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)

    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg)):
        sys.stdout.write(' ')
    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
