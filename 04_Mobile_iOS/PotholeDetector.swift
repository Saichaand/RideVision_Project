import Vision
import CoreML
import UIKit

/// Wraps a YOLOv8n CoreML pothole model exported from the training notebook
/// (best.mlpackage). Uses Vision's built-in YOLO postprocessing (NMS,
/// confidence filtering) via VNCoreMLModel + VNRecognizedObjectObservation.
///
/// Usage:
///   let detector = try PotholeDetector()
///   detector.detect(in: cgImage) { detections in
///       detections.forEach { print("box=\($0.boundingBox) conf=\($0.confidence)") }
///   }
final class PotholeDetector {

    struct Detection {
        let boundingBox: CGRect   // normalized (0-1), origin bottom-left, per Vision convention
        let confidence: Float
    }

    private let visionModel: VNCoreMLModel
    private let confidenceThreshold: Float

    init(confidenceThreshold: Float = 0.45) throws {
        // Xcode auto-generates a class named after your .mlpackage file, e.g. "best"
        // Replace `best` below with the actual generated class name after adding
        // best.mlpackage to your Xcode project.
        let config = MLModelConfiguration()
        config.computeUnits = .all // uses Neural Engine when available

        let coreMLModel = try best(configuration: config).model
        self.visionModel = try VNCoreMLModel(for: coreMLModel)
        self.confidenceThreshold = confidenceThreshold
    }

    /// Runs detection on a single frame (e.g. a CVPixelBuffer from camera output,
    /// converted to CGImage, or any CGImage from a captured photo).
    func detect(in image: CGImage, completion: @escaping ([Detection]) -> Void) {
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            guard let self else { return }

            guard error == nil,
                  let results = request.results as? [VNRecognizedObjectObservation] else {
                completion([])
                return
            }

            let detections = results
                .filter { $0.confidence >= self.confidenceThreshold }
                .map { Detection(boundingBox: $0.boundingBox, confidence: $0.confidence) }

            completion(detections)
        }

        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            try? handler.perform([request])
        }
    }
}

/*
Setup steps in Xcode:

1. Drag best.mlpackage into your Xcode project (check "Copy items if needed").
2. Xcode auto-generates a Swift class from it — the name matches the file
   (e.g. best.mlpackage -> class `best`). Update the `best(configuration:)`
   call above if your file/class name differs.
3. Vision handles YOLO's raw output -> bounding boxes + confidence + NMS
   automatically here, since CoreML export from Ultralytics includes the
   necessary metadata for VNRecognizedObjectObservation.

Converting a camera frame (CVPixelBuffer) to CGImage for the detect() call:

    import CoreImage

    func cgImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        return context.createCGImage(ciImage, from: ciImage.extent)
    }

Multi-frame confirmation (recommended, from the design discussion):
Track detections per rough location bucket (e.g. round GPS to ~5m grid) across
consecutive frames during a trip. Only surface / auto-file a detection once it
has appeared in 3+ consecutive frames above threshold, to filter shadow/manhole
false positives before they reach the report queue.
*/
