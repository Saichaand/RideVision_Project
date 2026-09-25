package com.ridevision.app.api

import com.google.gson.annotations.SerializedName
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST
import java.util.concurrent.TimeUnit

data class CheckAheadRequest(
    val lat: Double,
    val lon: Double,
    val heading: Float,
    @SerializedName("alert_radius_m") val alertRadiusM: Float = 250f
)

data class WarningInfo(
    @SerializedName("pothole_id") val potholeId: String,
    @SerializedName("distance_m") val distanceM: Int,
    @SerializedName("angular_deviation_deg") val deviationDeg: Float,
    val severity: String,
    val urgency: String,
    val address: String,
    val message: String
)

data class CheckAheadResponse(
    @SerializedName("has_warning") val hasWarning: Boolean,
    val warning: WarningInfo?
)

data class PotholeReportRequest(
    val lat: Double,
    val lon: Double,
    val severity: String,
    val address: String?,
    val note: String?,
    @SerializedName("user_id") val userId: String = "android-commuter"
)

data class MunicipalRouting(
    val city: String,
    val authority: String,
    val channel: String,
    val link: String?,
    val contact: String?,
    val instructions: String?
)

data class PotholeReportResponse(
    val success: Boolean,
    @SerializedName("is_new_hazard") val isNewHazard: Boolean,
    @SerializedName("report_id") val reportId: String,
    @SerializedName("municipal_routing") val municipalRouting: MunicipalRouting
)

interface RideVisionApiService {
    @POST("api/trip/check-ahead")
    suspend fun checkWarningAhead(@Body request: CheckAheadRequest): Response<CheckAheadResponse>

    @POST("api/potholes/report")
    suspend fun submitReport(@Body request: PotholeReportRequest): Response<PotholeReportResponse>
}

object RideVisionApiClient {
    // 10.0.2.2 maps to host machine localhost in Android Studio Emulator
    private const val BASE_URL = "http://10.0.2.2:8000/"

    val service: RideVisionApiService by lazy {
        val logging = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        }
        val client = OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(10, TimeUnit.SECONDS)
            .addInterceptor(logging)
            .build()

        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .client(client)
            .build()
            .create(RideVisionApiService::class.java)
    }
}
