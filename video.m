#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) return 2;
        NSURL *input = [NSURL fileURLWithPath:@(argv[1])];
        NSString *directory = @(argv[2]);
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:input options:nil];
        double seconds = CMTimeGetSeconds(asset.duration);
        if (!isfinite(seconds) || seconds <= 0 || [asset tracksWithMediaType:AVMediaTypeVideo].count == 0) return 3;

        AVAssetImageGenerator *generator = [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
        generator.appliesPreferredTrackTransform = YES;
        generator.maximumSize = CGSizeMake(960, 960);
        generator.requestedTimeToleranceBefore = CMTimeMake(1, 4);
        generator.requestedTimeToleranceAfter = CMTimeMake(1, 4);

        double interval = MAX(1, ceil(seconds / 30));
        NSMutableArray *frames = [NSMutableArray array];
        for (int index = 0; index < 30 && index * interval < seconds; index++) {
            @autoreleasepool {
                CMTime actual = kCMTimeInvalid;
                CMTime requested = CMTimeMakeWithSeconds(index * interval, 600);
                CGImageRef frame = [generator copyCGImageAtTime:requested actualTime:&actual error:NULL];
                if (!frame) {
                    [frames addObject:@{@"timestamp": @(index * interval), @"decode_failed": @YES}];
                    continue;
                }
                NSString *path = [directory stringByAppendingPathComponent:[NSString stringWithFormat:@"frame-%02d.jpg", index]];
                NSURL *output = [NSURL fileURLWithPath:path];
                CGImageDestinationRef destination = CGImageDestinationCreateWithURL(
                    (__bridge CFURLRef)output, CFSTR("public.jpeg"), 1, NULL);
                BOOL written = NO;
                if (destination) {
                    CGImageDestinationAddImage(destination, frame, NULL);
                    written = CGImageDestinationFinalize(destination);
                    CFRelease(destination);
                }
                CGImageRelease(frame);
                if (!written) return 1;
                double timestamp = CMTimeGetSeconds(actual);
                [frames addObject:@{@"path": path, @"timestamp": @(isfinite(timestamp) ? timestamp : index * interval)}];
            }
        }

        if (frames.count == 0) return 1;
        NSData *json = [NSJSONSerialization dataWithJSONObject:@{@"frames":frames} options:0 error:NULL];
        if (!json) return 1;
        fwrite(json.bytes, 1, json.length, stdout);
    }
    return 0;
}
