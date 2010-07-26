# -*- coding: utf-8 -*-

import math
import sys

import numpy
from PIL import Image, ImageDraw
import pyffmpeg

class StatsObserver(object):
    def __init__(self):
        self.sample_count = 0
        self.maximum_level = 0

    def __call__(self, data):
        samples = samples_to_mono(data[0])
        self.maximum_level = max(self.maximum_level, get_maximum_level(samples))
        self.sample_count += len(samples)
        del data, samples


class PeakObserver(object):
    def __init__(self, sample_count, length, delay=20):
        self.sample_count = sample_count
        self.length = length
        self.delay = delay

        self.samples_per_line = float(sample_count) / length
        self.buffered_len = 0
        self.buffered = []
        self.processed = 0.0
        self.j = 0
        self.peaks_upper = []
        self.peaks_lower = []
        self.peaks = (self.peaks_upper, self.peaks_lower)

    def get_observer_data(self, data):
        raise NotImplementedError

    def get_peaks(self, sliced):
        raise NotImplementedError

    def __call__(self, data):
        samples = self.get_observer_data(data)
        del data
        self.buffered.append(samples)
        self.buffered_len += len(samples)
        if self.buffered_len > self.delay * self.samples_per_line:
            self.process_peaks()
            self.buffered_len = 0

    def process_peaks(self, force=False):
        buffered = self.buffered
        samples = numpy.concatenate(buffered)
        i = 0
        length = len(samples)

        while True:
            new_processed = self.processed + self.samples_per_line
            to_process = int(round(new_processed) - round(self.processed))
            j = i + to_process
            if not force and j >= length:
                break
            sliced = samples[i:j]
            self.j += len(sliced)

            if len(sliced):
                upper, lower = self.get_peaks(sliced)
                self.peaks_upper.append(upper)
                self.peaks_lower.append(lower)
                del sliced

            i = j
            self.processed = new_processed
            if i > length:
                break

        if force:
            self.buffered = []
        else:
            self.buffered = [samples[i:]]
        del buffered, samples

    def finish(self):
        self.process_peaks(True)


class PeakObserverMinMax(PeakObserver):
    '''
    Above: Maximum level
    Below: Minimum level
    '''
    def get_observer_data(self, data):
        return samples_to_mono(data[0])

    def get_peaks(self, sliced):
        upper = numpy.max(sliced)
        lower = numpy.min(sliced)
        return (upper, lower)


class PeakObserverStereo(PeakObserver):
    '''
    Above: Left channel
    Below: Right channel
    '''
    def get_observer_data(self, data):
        samples = numpy.abs(data[0])
        assert len(samples[0]) == 2
        return samples

    def get_peaks(self, sliced):
        upper, lower = numpy.max(sliced, axis=0)
        lower = -lower
        return (upper, lower)


class SpectrumObserver(object):
    def __init__(self):
        pass

    def __call__(self, data):
        samples = samples_to_mono(data[0])


def get_maximum_level(samples):
    return max(abs(samples.min()), abs(samples.max()))

def samples_to_mono(samples):
    samples = numpy.average(samples, axis=1)
    samples2 = samples.astype(numpy.int16)
    del samples
    return samples2

def observe_file(filename, observer, track_no=0):
    reader = pyffmpeg.FFMpegReader()
    reader.open(filename, track_selector=pyffmpeg.TS_AUDIO)
    track = reader.get_tracks()[track_no]

    track.set_observer(observer)
    try:
        reader.run()
    except IOError:
        pass
    reader.close()

def draw_line(image, draw, x, y1, y2):
    y1a = int(math.ceil(y1))
    y1b = int(round(255 * (y1a - y1)))
    y2a = int(math.floor(y2))
    y2b = int(round(255 * (y2 - y2a)))

    draw.line((x, y1a, x, y2a), 255)
    if y1b:
        try:
            image.putpixel((x, y1a-1), y1b)
        except:
            pass
    if y2b:
        try:
            image.putpixel((x, y2a+1), y2b)
        except:
            pass

def scale_peaks(peaks, half):
    scale = float(-half) / (2**15)
    scaled = numpy.multiply(peaks, scale)
    scaled += half
    return scaled

def average_peaks(peaks, count):
    if not (count and count > 1):
        return peaks
    assert len(peaks[0]) % count == 0
    upper, lower = peaks
    upper = numpy.reshape(upper, (-1, count))
    lower = numpy.reshape(lower, (-1, count))
    upper = numpy.average(upper, axis = 1)
    lower = numpy.average(lower, axis = 1)
    peaks = numpy.array((upper, lower)).astype(numpy.int16)
    return peaks

def max_peaks(peaks, count):
    if not (count and count > 1):
        return peaks
    assert len(peaks[0]) % count == 0
    upper, lower = peaks
    upper = numpy.reshape(upper, (-1, count))
    lower = numpy.reshape(lower, (-1, count))
    upper = numpy.amax(upper, axis = 1)
    lower = numpy.amin(lower, axis = 1)
    peaks = numpy.array((upper, lower)).astype(numpy.int16)
    return peaks

def draw_peaks(peaks, image, image_draw, offset=0, jump=1):
    x = 0
    upper, lower = peaks
    for i in xrange(offset, len(upper), jump):
        y1, y2 = upper[i], lower[i]
        draw_line(image, image_draw, x, y1, y2)
        x += 1

def render_waveform(peaks, half, average=None, antialias=None):
    height = half*2 + 1
    width = len(peaks[0])

    if average and average > 1:
        peaks = average_peaks(peaks, average)
        width /= average

    if antialias and antialias > 1:
        assert width % antialias == 0
        width /= antialias
    else:
        antialias = 0

    image = Image.new('L', (width, height), 0)
    image_draw = ImageDraw.Draw(image)

    scaled = scale_peaks(peaks, half)

    if antialias:
        output = numpy.zeros((height, width), numpy.uint32)
        for i in xrange(antialias):
            image_draw.rectangle(((0, 0), (width, height)), fill = 0)
            draw_peaks(scaled, image, image_draw, i, antialias)
            output += numpy.asarray(image).astype(numpy.uint32)
        output /= antialias
        image = Image.fromarray(output.astype(numpy.uint8))
    else:
        draw_peaks(scaled, image, image_draw)
    return image

def main():
    multi = 20
    width = 700

    a = int(multi**0.5)
    b = multi / a
    multi = a * b

    assert len(sys.argv) == 3
    print 'Reading statistics'
    # Because I can't find reliable method to get sample count...
    # It's slow but... worse is better
    o_stats = StatsObserver()
    observe_file(sys.argv[1], o_stats)
    maximum_level = o_stats.maximum_level
    sample_count = o_stats.sample_count

    print 'Reading peaks'
    o_peaks = PeakObserverStereo(sample_count, width*multi)
    observe_file(sys.argv[1], o_peaks)
    o_peaks.finish()

    upper, lower = o_peaks.peaks
    length = len(upper)
    upper = upper[:length-(length%multi)]
    lower = lower[:length-(length%multi)]
    data = (upper, lower)

    #image = render_waveform(data, 48)
    image = render_waveform(max_peaks(data, multi), 48)
    image.save(sys.argv[2] + '.png')
    image = render_waveform(data, 48, average=multi)
    image.save(sys.argv[2] + '_avg.png')
    image = render_waveform(data, 48, antialias=multi)
    image.save(sys.argv[2] + '_aa.png')
    image = render_waveform(data, 48, average=a, antialias=b)
    image.save(sys.argv[2] + '_hyb.png')

if __name__ == '__main__':
    main()
