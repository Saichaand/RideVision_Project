import Foundation
import CoreLocation

struct WarningData: Decodable {
    let potholeId: String
    let distanceM: Int
    let severity: String
    let address: String
    let message: String

    enum CodingKeys: String, CodingKey {
        case potholeId = "pothole_id"
        case distanceM = "distance_m"
        case severity
        case address
        case message
    }
}

final class LocationManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    @Published var lastLocation: CLLocation?
    @Published var currentWarning: WarningData?

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBestForNavigation
        manager.distanceFilter = 5.0
        manager.requestWhenInUseAuthorization()
        manager.startUpdatingLocation()
        manager.startUpdatingHeading()
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let loc = locations.last else { return }
        self.lastLocation = loc

        let heading = loc.course >= 0 ? loc.course : 0.0

        NetworkService.shared.checkWarningAhead(lat: loc.coordinate.latitude, lon: loc.coordinate.longitude, heading: Float(heading)) { [weak self] warning in
            DispatchQueue.main.async {
                self?.currentWarning = warning
            }
        }
    }
}
