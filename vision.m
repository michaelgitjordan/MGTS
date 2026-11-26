#import <Foundation/Foundation.h>
#import <Vision/Vision.h>

// Only pixels from the supplied crop reach the feature extractor. Paths are
// temporary transport names and have no part in classification.
int main(void) {
    @autoreleasepool {
        NSData *input = [[NSFileHandle fileHandleWithStandardInput] readDataToEndOfFile];
        NSArray *paths = [NSJSONSerialization JSONObjectWithData:input options:0 error:NULL];
        if (![paths isKindOfClass:NSArray.class]) return 2;
        for (NSString *path in paths) {
            @autoreleasepool {
                VNGenerateImageFeaturePrintRequest *request = [VNGenerateImageFeaturePrintRequest new];
                if (@available(macOS 14.0, *)) request.revision = VNGenerateImageFeaturePrintRequestRevision2;
                request.usesCPUOnly = YES;
                request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFit;
                VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithURL:[NSURL fileURLWithPath:path] options:@{}];
                NSError *error = nil;
                if (![handler performRequests:@[request] error:&error]) {
                    fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
                    return 1;
                }
                VNFeaturePrintObservation *feature = request.results.firstObject;
                if (!feature || feature.elementType != VNElementTypeFloat) return 1;
                NSString *encoded = [feature.data base64EncodedStringWithOptions:0];
                puts(encoded.UTF8String);
            }
        }
    }
    return 0;
}
