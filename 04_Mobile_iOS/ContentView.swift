import SwiftUI
import Vision
import CoreLocation

struct ContentView: View {
    @StateObject private var locationManager = LocationManager()
    @State private var selectedImage: UIImage?
    @State private var detections: [PotholeDetector.Detection] = []
    @State private var isShowingImagePicker = false
    @State private var alertMessage: String?
    @State private var showAlert = false

    var body: some View {
        NavigationView {
            ZStack {
                Color(UIColor.systemBackground).edgesIgnoringSafeArea(.all)

                VStack(spacing: 0) {
                    // Header Status
                    HStack {
                        VStack(alignment: .leading) {
                            Text("RideVision iOS")
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(.red)
                            Text("Real-Time CV Hazard Alert")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Label("GPS Active", systemImage: "location.fill")
                            .font(.caption)
                            .padding(6)
                            .background(Color.green.opacity(0.2))
                            .foregroundColor(.green)
                            .cornerRadius(6)
                    }
                    .padding()

                    // Proactive Warning Ahead Banner
                    if let warning = locationManager.currentWarning {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundColor(.red)
                                Text("HAZARD AHEAD (\(warning.distanceM)m)")
                                    .font(.headline)
                                    .foregroundColor(.red)
                            }
                            Text(warning.message)
                                .font(.subheadline)
                                .foregroundColor(.primary)
                            Text(warning.address)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.15))
                        .cornerRadius(10)
                        .padding(.horizontal)
                        .transition(.slide)
                    }

                    // Viewfinder / Image Display
                    ZStack {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(UIColor.secondarySystemBackground))

                        if let image = selectedImage {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                .overlay(
                                    GeometryReader { geo in
                                        ForEach(0..<detections.count, id: \.self) { i in
                                            let d = detections[i]
                                            let rect = normalizeVisionRect(d.boundingBox, in: geo.size)
                                            Rectangle()
                                                .stroke(Color.red, lineWidth: 3)
                                                .frame(width: rect.width, height: rect.height)
                                                .position(x: rect.midX, y: rect.midY)
                                        }
                                    }
                                )
                        } else {
                            VStack(spacing: 8) {
                                Image(systemName: "camera.viewfinder")
                                    .font(.system(size: 48))
                                    .foregroundColor(.secondary)
                                Text("Select photo or point camera\nto detect potholes with CoreML")
                                    .font(.footnote)
                                    .multilineTextAlignment(.center)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding()

                    // Controls
                    HStack(spacing: 16) {
                        Button(action: { isShowingImagePicker = true }) {
                            Label("Pick Photo", systemImage: "photo.on.rectangle")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }

                        Button(action: submitHazardReport) {
                            Label("Report Hazard", systemImage: "paperplane.fill")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.red)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                    .padding()
                }
            }
            .navigationBarHidden(true)
            .sheet(isPresented: $isShowingImagePicker) {
                ImagePicker(image: $selectedImage, onPicked: runDetection)
            }
            .alert(isPresented: $showAlert) {
                Alert(title: Text("RideVision"), message: Text(alertMessage ?? ""), dismissButton: .default(Text("OK")))
            }
        }
    }

    private func runDetection(image: UIImage) {
        guard let cgImage = image.cgImage else { return }
        do {
            let detector = try PotholeDetector()
            detector.detect(in: cgImage) { results in
                DispatchQueue.main.async {
                    self.detections = results
                }
            }
        } catch {
            print("CoreML detector error: \(error)")
        }
    }

    private func normalizeVisionRect(_ box: CGRect, in size: CGSize) -> CGRect {
        // Vision coordinate origin is bottom-left
        let x = box.origin.x * size.width
        let y = (1 - box.origin.y - box.height) * size.height
        let w = box.width * size.width
        let h = box.height * size.height
        return CGRect(x: x, y: y, width: w, height: h)
    }

    private func submitHazardReport() {
        guard let loc = locationManager.lastLocation else {
            alertMessage = "Waiting for GPS fix..."
            showAlert = true
            return
        }

        NetworkService.shared.submitReport(lat: loc.coordinate.latitude, lon: loc.coordinate.longitude, severity: "severe") { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let res):
                    alertMessage = "Report filed! Municipal channel: \(res.routing.authority)"
                    if let waLink = res.routing.link, let url = URL(string: waLink), res.routing.channel == "whatsapp" {
                        UIApplication.shared.open(url)
                    }
                case .failure(let err):
                    alertMessage = "Report error: \(err.localizedDescription)"
                }
                showAlert = true
            }
        }
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    var onPicked: (UIImage) -> Void
    @Environment(\.presentationMode) var presentationMode

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker
        init(_ parent: ImagePicker) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
                parent.onPicked(uiImage)
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}
