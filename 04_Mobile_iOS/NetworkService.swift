import Foundation

struct CheckAheadResponsePayload: Decodable {
    let hasWarning: Bool
    let warning: WarningData?

    enum CodingKeys: String, CodingKey {
        case hasWarning = "has_warning"
        case warning
    }
}

struct ReportResponsePayload: Decodable {
    let success: Bool
    let reportId: String
    let routing: MunicipalRoutingPayload

    enum CodingKeys: String, CodingKey {
        case success
        case reportId = "report_id"
        case routing = "municipal_routing"
    }
}

struct MunicipalRoutingPayload: Decodable {
    let city: String
    let authority: String
    let channel: String
    let link: String?
}

final class NetworkService {
    static let shared = NetworkService()
    private let baseURL = "http://localhost:8000"

    func checkWarningAhead(lat: Double, lon: Double, heading: Float, completion: @escaping (WarningData?) -> Void) {
        guard let url = URL(string: "\(baseURL)/api/trip/check-ahead") else { return }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "lat": lat,
            "lon": lon,
            "heading": heading,
            "alert_radius_m": 250
        ]
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)

        URLSession.shared.dataTask(with: req) { data, _, _ in
            guard let data = data,
                  let decoded = try? JSONDecoder().decode(CheckAheadResponsePayload.self, from: data) else {
                completion(nil)
                return
            }
            completion(decoded.warning)
        }.resume()
    }

    func submitReport(lat: Double, lon: Double, severity: String, completion: @escaping (Result<ReportResponsePayload, Error>) -> Void) {
        guard let url = URL(string: "\(baseURL)/api/potholes/report") else { return }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "lat": lat,
            "lon": lon,
            "severity": severity,
            "user_id": "ios-commuter"
        ]
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)

        URLSession.shared.dataTask(with: req) { data, _, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            guard let data = data,
                  let decoded = try? JSONDecoder().decode(ReportResponsePayload.self, from: data) else {
                completion(.failure(NSError(domain: "DecodeError", code: 0)))
                return
            }
            completion(.success(decoded))
        }.resume()
    }
}
