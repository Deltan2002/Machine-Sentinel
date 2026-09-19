#include <Wire.h>
#include <math.h>

// ==================================================
// MPU6050 configuration
// ==================================================

constexpr uint8_t MPU_ADDR = 0x68;

constexpr uint8_t WHO_AM_I     = 0x75;
constexpr uint8_t PWR_MGMT_1   = 0x6B;
constexpr uint8_t ACCEL_CONFIG = 0x1C;
constexpr uint8_t ACCEL_XOUT_H = 0x3B;

// ==================================================
// Sampling configuration
// ==================================================

constexpr int kSampleRate = 200;
constexpr int kWindowSize = 200;

constexpr uint32_t kSamplePeriodUs = 5000;

// ±8g sensitivity
constexpr float kCountsPerG = 4096.0f;

// ==================================================
// Trained edge-model constants
// ==================================================

// StandardScaler means
constexpr float kRmsMean = 0.2503907709f;
constexpr float kPtpMean = 0.3384182347f;

// StandardScaler scales
constexpr float kRmsScale = 0.1136292923f;
constexpr float kPtpScale = 0.1760807222f;

// Logistic Regression weights
constexpr float kRmsWeight = 2.3543119820f;
constexpr float kPtpWeight = 2.1076915519f;

// Logistic Regression intercept
constexpr float kIntercept = 3.4872158012f;

// ==================================================
// Decision thresholds
// ==================================================

// Prototype activity threshold
constexpr float kIdleThreshold = 0.04637709f;

// Standard binary Logistic Regression threshold
constexpr float kAnomalyThreshold = 0.5f;

// ==================================================
// Sample buffers
// ==================================================

int16_t accelX[kWindowSize];
int16_t accelY[kWindowSize];
int16_t accelZ[kWindowSize];

int sampleIndex = 0;

uint32_t previousSampleTime = 0;

// ==================================================
// Feature structure
// ==================================================

struct Features
{
    float rms;
    float ptp;
};

// ==================================================
// State
// ==================================================

enum class MachineState
{
    Idle,
    Normal,
    Anomaly
};

// ==================================================
// Write MPU register
// ==================================================

bool writeRegister(
    uint8_t deviceAddress,
    uint8_t registerAddress,
    uint8_t value
)
{
    Wire.beginTransmission(deviceAddress);

    Wire.write(registerAddress);
    Wire.write(value);

    const uint8_t error =
        Wire.endTransmission();

    return error == 0;
}

// ==================================================
// Read one MPU register
// ==================================================

bool readRegister(
    uint8_t deviceAddress,
    uint8_t registerAddress,
    uint8_t& value
)
{
    Wire.beginTransmission(deviceAddress);

    Wire.write(registerAddress);

    if (Wire.endTransmission(false) != 0)
    {
        return false;
    }

    Wire.requestFrom(
        deviceAddress,
        static_cast<uint8_t>(1)
    );

    if (Wire.available() < 1)
    {
        return false;
    }

    value = Wire.read();

    return true;
}

// ==================================================
// Read consecutive MPU registers
// ==================================================

bool readRegisters(
    uint8_t deviceAddress,
    uint8_t startRegister,
    uint8_t* buffer,
    uint8_t length
)
{
    Wire.beginTransmission(deviceAddress);

    Wire.write(startRegister);

    if (Wire.endTransmission(false) != 0)
    {
        return false;
    }

    Wire.requestFrom(
        deviceAddress,
        length
    );

    if (Wire.available() < length)
    {
        return false;
    }

    for (uint8_t i = 0; i < length; ++i)
    {
        buffer[i] = Wire.read();
    }

    return true;
}

// ==================================================
// Read raw acceleration
// ==================================================

bool readAccelerationRaw(
    int16_t& x,
    int16_t& y,
    int16_t& z
)
{
    uint8_t data[6];

    if (!readRegisters(
            MPU_ADDR,
            ACCEL_XOUT_H,
            data,
            6))
    {
        return false;
    }

    x = static_cast<int16_t>(
        (static_cast<uint16_t>(data[0]) << 8)
        | data[1]
    );

    y = static_cast<int16_t>(
        (static_cast<uint16_t>(data[2]) << 8)
        | data[3]
    );

    z = static_cast<int16_t>(
        (static_cast<uint16_t>(data[4]) << 8)
        | data[5]
    );

    return true;
}

// ==================================================
// Calculate RMS + P2P
// ==================================================

Features calculateFeatures()
{
    float sumX = 0.0f;
    float sumY = 0.0f;
    float sumZ = 0.0f;

    // ----------------------------------------------
    // Pass 1: calculate mean of each axis
    // ----------------------------------------------

    for (int i = 0; i < kWindowSize; ++i)
    {
        sumX += accelX[i];
        sumY += accelY[i];
        sumZ += accelZ[i];
    }

    const float meanX =
        sumX / kWindowSize;

    const float meanY =
        sumY / kWindowSize;

    const float meanZ =
        sumZ / kWindowSize;

    // ----------------------------------------------
    // Pass 2: calculate dynamic vibration
    // ----------------------------------------------

    float sumMagnitudeSquared = 0.0f;

    float minMagnitude = INFINITY;
    float maxMagnitude = -INFINITY;

    for (int i = 0; i < kWindowSize; ++i)
    {
        // Remove constant gravity / DC component
        const float dx =
            accelX[i] - meanX;

        const float dy =
            accelY[i] - meanY;

        const float dz =
            accelZ[i] - meanZ;

        const float magnitudeSquared =
            dx * dx +
            dy * dy +
            dz * dz;

        sumMagnitudeSquared +=
            magnitudeSquared;

        const float magnitude =
            sqrtf(magnitudeSquared);

        if (magnitude < minMagnitude)
        {
            minMagnitude = magnitude;
        }

        if (magnitude > maxMagnitude)
        {
            maxMagnitude = magnitude;
        }
    }

    // ----------------------------------------------
    // RMS
    // ----------------------------------------------

    const float rmsCounts =
        sqrtf(
            sumMagnitudeSquared
            / kWindowSize
        );

    const float rms =
        rmsCounts
        / kCountsPerG;

    // ----------------------------------------------
    // P2P
    // ----------------------------------------------

    const float ptp =
        (
            maxMagnitude
            -
            minMagnitude
        )
        / kCountsPerG;

    return {
        rms,
        ptp
    };
}

// ==================================================
// Logistic Regression inference
// ==================================================

float predictAnomalyProbability(
    float rms,
    float ptp
)
{
    // Same StandardScaler transformation as Python
    const float scaledRms =
        (rms - kRmsMean)
        / kRmsScale;

    const float scaledPtp =
        (ptp - kPtpMean)
        / kPtpScale;

    // Logistic Regression linear score
    const float score =
        kRmsWeight * scaledRms
        +
        kPtpWeight * scaledPtp
        +
        kIntercept;

    // Sigmoid
    const float probability =
        1.0f
        /
        (
            1.0f
            +
            expf(-score)
        );

    return probability;
}

// ==================================================
// Decide machine state
// ==================================================

MachineState determineState(
    const Features& features,
    float& anomalyProbability
)
{
    // ----------------------------------------------
    // First determine whether machine is active
    // ----------------------------------------------

    if (features.rms < kIdleThreshold)
    {
        anomalyProbability = 0.0f;

        return MachineState::Idle;
    }

    // ----------------------------------------------
    // Machine is active:
    // run ML classifier
    // ----------------------------------------------

    anomalyProbability =
        predictAnomalyProbability(
            features.rms,
            features.ptp
        );

    if (
        anomalyProbability
        >= kAnomalyThreshold
    )
    {
        return MachineState::Anomaly;
    }

    return MachineState::Normal;
}

// ==================================================
// Print state
// ==================================================

void printState(
    MachineState state
)
{
    switch (state)
    {
        case MachineState::Idle:
            Serial.print("IDLE");
            break;

        case MachineState::Normal:
            Serial.print("NORMAL");
            break;

        case MachineState::Anomaly:
            Serial.print("ANOMALY");
            break;
    }
}

// ==================================================
// Setup
// ==================================================

void setup()
{
    Serial.begin(115200);

    delay(1000);

    Wire.begin();

    Serial.println();
    Serial.println(
        "================================"
    );

    Serial.println(
        "Machine Sentinel"
    );

    Serial.println(
        "Edge Vibration Anomaly Detector"
    );

    Serial.println(
        "================================"
    );

    // ----------------------------------------------
    // WHO_AM_I
    // ----------------------------------------------

    uint8_t whoAmI = 0;

    if (!readRegister(
            MPU_ADDR,
            WHO_AM_I,
            whoAmI))
    {
        Serial.println(
            "ERROR: MPU6050 not responding"
        );

        while (true)
        {
        }
    }

    Serial.print(
        "WHO_AM_I: 0x"
    );

    Serial.println(
        whoAmI,
        HEX
    );

    // ----------------------------------------------
    // Wake MPU
    // ----------------------------------------------

    if (!writeRegister(
            MPU_ADDR,
            PWR_MGMT_1,
            0x01))
    {
        Serial.println(
            "ERROR: Failed to wake MPU"
        );

        while (true)
        {
        }
    }

    delay(100);

    // ----------------------------------------------
    // ±8g accelerometer range
    // ----------------------------------------------

    if (!writeRegister(
            MPU_ADDR,
            ACCEL_CONFIG,
            0x10))
    {
        Serial.println(
            "ERROR: Failed to configure accelerometer"
        );

        while (true)
        {
        }
    }

    delay(100);

    // ----------------------------------------------
    // Verify range configuration
    // ----------------------------------------------

    uint8_t accelConfig = 0;

    if (readRegister(
            MPU_ADDR,
            ACCEL_CONFIG,
            accelConfig))
    {
        Serial.print(
            "ACCEL_CONFIG: 0x"
        );

        Serial.println(
            accelConfig,
            HEX
        );
    }

    Serial.println();
    Serial.println(
        "Sampling: 200 Hz"
    );

    Serial.println(
        "Window: 1 second"
    );

    Serial.println(
        "Features: RMS + P2P"
    );

    Serial.println(
        "Model: Logistic Regression"
    );

    Serial.println();

    // Important:
    // start sampling clock only after setup
    previousSampleTime =
        micros();
}

// ==================================================
// Main loop
// ==================================================

void loop()
{
    const uint32_t currentTime =
        micros();

    if (
        currentTime
        -
        previousSampleTime
        >=
        kSamplePeriodUs
    )
    {
        previousSampleTime +=
            kSamplePeriodUs;

        int16_t x;
        int16_t y;
        int16_t z;

        if (!readAccelerationRaw(
                x,
                y,
                z))
        {
            Serial.println(
                "ERROR: MPU read failed"
            );

            return;
        }

        // ------------------------------------------
        // Store this sample
        // ------------------------------------------

        accelX[sampleIndex] = x;
        accelY[sampleIndex] = y;
        accelZ[sampleIndex] = z;

        ++sampleIndex;

        // ------------------------------------------
        // 200 samples = one complete window
        // ------------------------------------------

        if (sampleIndex >= kWindowSize)
        {
            const Features features =
                calculateFeatures();

            float anomalyProbability =
                0.0f;

            const MachineState state =
                determineState(
                    features,
                    anomalyProbability
                );

            // --------------------------------------
            // Output
            // --------------------------------------

            Serial.print(
                "RMS: "
            );

            Serial.print(
                features.rms,
                6
            );

            Serial.print(
                " g | P2P: "
            );

            Serial.print(
                features.ptp,
                6
            );

            Serial.print(
                " g | P(anomaly): "
            );

            // IDLE probability is intentionally 0
            Serial.print(
                anomalyProbability,
                4
            );

            Serial.print(
                " | STATE: "
            );

            printState(state);

            Serial.println();

            // --------------------------------------
            // Start next 1-second window
            // --------------------------------------

            sampleIndex = 0;
        }
    }
}