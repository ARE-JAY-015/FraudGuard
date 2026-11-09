const API_URL = "http://127.0.0.1:8000";


/* =====================================
   ELEMENTS
===================================== */

const form =
    document.getElementById("transactionForm");

const analyzeBtn =
    document.getElementById("analyzeBtn");

const resetBtn =
    document.getElementById("resetBtn");

const loadDemoBtn =
    document.getElementById("loadDemo");

const formMessage =
    document.getElementById("formMessage");

const apiStatus =
    document.getElementById("apiStatus");


const emptyResult =
    document.getElementById("emptyResult");

const resultContent =
    document.getElementById("resultContent");

const resultBadge =
    document.getElementById("resultBadge");

const riskCard =
    document.getElementById("riskCard");

const meterFill =
    document.getElementById("meterFill");


const predictionText =
    document.getElementById("predictionText");

const probability =
    document.getElementById("probability");

const riskLevel =
    document.getElementById("riskLevel");

const resultAmount =
    document.getElementById("resultAmount");

const historyRatio =
    document.getElementById("historyRatio");

const resultFailed =
    document.getElementById("resultFailed");

const reasonsList =
    document.getElementById("reasonsList");


const historyBody =
    document.getElementById("historyBody");

const historyCount =
    document.getElementById("historyCount");


/* =====================================
   GET VALUES
===================================== */

function value(id) {

    return document
        .getElementById(id)
        .value;
}


function numberValue(id) {

    return Number(
        document
            .getElementById(id)
            .value
    );
}


/* =====================================
   API STATUS
===================================== */

function setApiStatus(
    online,
    message
) {

    apiStatus.textContent = message;

    const dot =
        document.querySelector(
            ".api-pill .status-dot"
        );


    dot.style.background =
        online
            ? "#27c991"
            : "#dc3c4b";


    dot.style.boxShadow =
        online

            ? "0 0 0 4px rgba(39,201,145,.12)"

            : "0 0 0 4px rgba(220,60,75,.12)";
}


async function checkApi() {

    try {

        const response =
            await fetch(
                `${API_URL}/health`,
                {
                    method: "GET"
                }
            );


        if (!response.ok) {

            throw new Error(
                "API unavailable"
            );
        }


        setApiStatus(
            true,
            "API Connected"
        );

    }

    catch (error) {

        setApiStatus(
            false,
            "API Offline"
        );
    }
}


/* =====================================
   BUILD REQUEST
===================================== */

function getPayload() {

    return {

        customer_id:
            value("customer_id")
                .trim(),

        amount:
            numberValue("amount"),

        distance_from_home:
            numberValue(
                "distance_from_home"
            ),

        device_changed:
            document
                .getElementById(
                    "device_changed"
                )
                .checked
                ? 1
                : 0,

        international:
            document
                .getElementById(
                    "international"
                )
                .checked
                ? 1
                : 0,

        failed_attempts:
            numberValue(
                "failed_attempts"
            ),

        merchant_category:
            value(
                "merchant_category"
            ),

        payment_method:
            value(
                "payment_method"
            ),

        account_age_days:
            numberValue(
                "account_age_days"
            )
    };
}


/* =====================================
   VALIDATION
===================================== */

function validatePayload(data) {

    if (!data.customer_id) {

        return "Please enter a Customer ID.";
    }


    if (data.amount < 0) {

        return "Transaction amount cannot be negative.";
    }


    if (data.distance_from_home < 0) {

        return "Distance cannot be negative.";
    }


    if (data.failed_attempts < 0) {

        return "Failed attempts cannot be negative.";
    }


    if (data.account_age_days < 0) {

        return "Account age cannot be negative.";
    }


    return "";
}


/* =====================================
   MONEY
===================================== */

function formatMoney(amount) {

    return new Intl.NumberFormat(
        "en-IN",
        {
            style: "currency",
            currency: "INR",
            maximumFractionDigits: 2
        }
    ).format(
        Number(amount || 0)
    );
}


/* =====================================
   RISK CLASS
===================================== */

function riskClass(level) {

    const normalized =
        String(level || "")
            .toLowerCase();


    if (normalized === "high") {

        return "high";
    }


    if (normalized === "medium") {

        return "medium";
    }


    return "low";
}


/* =====================================
   DISPLAY RESULT
===================================== */

function displayResult(data) {

    const prediction =
        data.prediction || {};

    const behavior =
        data.behavior_analysis || {};


    const isFraud =
        Boolean(
            prediction.fraud
        );


    const risk =
        String(
            prediction.risk_level ||
            "LOW"
        ).toUpperCase();


    const probabilityValue =
        Number(
            prediction.fraud_probability ||
            0
        );


    const probabilityPercent =
        Math.max(
            0,
            Math.min(
                100,
                probabilityValue * 100
            )
        );


    const cls =
        riskClass(risk);


    /* Show result */

    emptyResult.classList.add(
        "hidden"
    );

    resultContent.classList.remove(
        "hidden"
    );


    /* Badge */

    resultBadge.textContent =
        risk;

    resultBadge.className =
        `result-badge ${cls}`;


    /* Risk card */

    riskCard.className =
        `risk-card ${cls}`;


    /* Prediction */

    predictionText.textContent =
        isFraud

            ? "FRAUD DETECTED"

            : "LIKELY LEGITIMATE";


    /* Probability */

    probability.textContent =
        probabilityPercent.toFixed(2);


    /* Risk */

    riskLevel.textContent =
        risk;


    /* Amount */

    resultAmount.textContent =
        formatMoney(
            behavior.amount ??
            data.transaction?.amount
        );


    /* History ratio */

    historyRatio.textContent =

        behavior.amount_ratio_to_history != null

            ? `${Number(
                behavior.amount_ratio_to_history
              ).toFixed(2)}×`

            : "—";


    /* Failed attempts */

    resultFailed.textContent =
        behavior.failed_attempts ??
        "—";


    /* Meter */

    meterFill.style.width =
        `${probabilityPercent}%`;


    meterFill.className =
        `meter-fill ${cls}`;


    /* Reasons */

    const reasons =

        Array.isArray(
            behavior.reasons
        )
        &&
        behavior.reasons.length

            ? behavior.reasons

            : [
                "No behavioral explanation was returned by the API."
            ];


    reasonsList.innerHTML =
        reasons

            .map(
                reason =>
                    `<li>
                        ${escapeHtml(reason)}
                    </li>`
            )

            .join("");


    /* History */

    renderHistory(
        data.customer_history || []
    );
}


/* =====================================
   HISTORY
===================================== */

function renderHistory(history) {

    historyCount.textContent =
        `${history.length} transaction${
            history.length === 1
                ? ""
                : "s"
        }`;


    if (!history.length) {

        historyBody.innerHTML =

            `<tr>
                <td
                    colspan="4"
                    class="table-empty"
                >
                    No customer history returned.
                </td>
            </tr>`;

        return;
    }


    historyBody.innerHTML =

        history

            .map(tx => {

                const fraud =
                    Number(tx.fraud) === 1;


                return `

                    <tr>

                        <td>
                            <strong>
                                ${escapeHtml(
                                    String(
                                        tx.transaction_id
                                    )
                                )}
                            </strong>
                        </td>


                        <td>
                            ${escapeHtml(
                                String(
                                    tx.timestamp ||
                                    "—"
                                )
                            )}
                        </td>


                        <td>
                            ${formatMoney(
                                tx.amount
                            )}
                        </td>


                        <td>

                            <span
                                class="${
                                    fraud
                                        ? "fraud-status"
                                        : "safe-status"
                                }"
                            >

                                ${
                                    fraud
                                        ? "FRAUD"
                                        : "LEGITIMATE"
                                }

                            </span>

                        </td>

                    </tr>

                `;
            })

            .join("");
}


/* =====================================
   SECURITY
===================================== */

function escapeHtml(text) {

    return String(text)

        .replaceAll(
            "&",
            "&amp;"
        )

        .replaceAll(
            "<",
            "&lt;"
        )

        .replaceAll(
            ">",
            "&gt;"
        )

        .replaceAll(
            '"',
            "&quot;"
        )

        .replaceAll(
            "'",
            "&#039;"
        );
}


/* =====================================
   ANALYZE TRANSACTION
===================================== */

async function analyzeTransaction(
    event
) {

    event.preventDefault();


    formMessage.textContent =
        "";


    const payload =
        getPayload();


    const validationError =
        validatePayload(
            payload
        );


    if (validationError) {

        formMessage.textContent =
            validationError;

        return;
    }


    /* Disable button */

    analyzeBtn.disabled =
        true;


    analyzeBtn
        .querySelector(
            "span:first-child"
        )
        .textContent =
        "Analyzing...";


    try {

        const response =
            await fetch(
                `${API_URL}/predict`,
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json",

                        "Accept":
                            "application/json"

                    },

                    body:
                        JSON.stringify(
                            payload
                        )
                }
            );


        let result;


        try {

            result =
                await response.json();

        }

        catch {

            throw new Error(
                `Server returned HTTP ${response.status}.`
            );
        }


        /* HTTP error */

        if (!response.ok) {

            const detail =

                Array.isArray(
                    result.detail
                )

                    ? result.detail
                        .map(
                            item =>
                                item.msg ||
                                "Validation error"
                        )
                        .join(", ")

                    : result.detail ||
                      `HTTP ${response.status}`;


            throw new Error(
                detail
            );
        }


        /* API error */

        if (!result.success) {

            throw new Error(
                "FraudGuard did not return a successful prediction."
            );
        }


        /* Display */

        displayResult(
            result
        );


        /* Scroll */

        document
            .getElementById(
                "results"
            )
            .scrollIntoView({

                behavior: "smooth",

                block: "start"

            });


    }

    catch (error) {

        formMessage.textContent =

            `Could not analyze transaction:
             ${error.message}.
             Make sure FastAPI is running
             on port 8000.`;
    }


    finally {

        analyzeBtn.disabled =
            false;


        analyzeBtn
            .querySelector(
                "span:first-child"
            )
            .textContent =
            "Analyze Transaction";
    }
}


/* =====================================
   RESET
===================================== */

function resetForm() {

    form.reset();


    document.getElementById(
        "customer_id"
    ).value = "";


    document.getElementById(
        "amount"
    ).value = "";


    document.getElementById(
        "account_age_days"
    ).value = "";


    document.getElementById(
        "distance_from_home"
    ).value = "";


    document.getElementById(
        "failed_attempts"
    ).value = "0";


    document.getElementById(
        "device_changed"
    ).checked = false;


    document.getElementById(
        "international"
    ).checked = false;


    formMessage.textContent =
        "";


    emptyResult.classList.remove(
        "hidden"
    );


    resultContent.classList.add(
        "hidden"
    );


    resultBadge.textContent =
        "WAITING";


    resultBadge.className =
        "result-badge neutral";


    historyBody.innerHTML =

        `<tr>

            <td
                colspan="4"
                class="table-empty"
            >
                History will appear after an analysis.
            </td>

        </tr>`;


    historyCount.textContent =
        "0 transactions";
}


/* =====================================
   DEMO TRANSACTION
===================================== */

function loadDemo() {

    document.getElementById(
        "customer_id"
    ).value =
        "CUST0983";


    document.getElementById(
        "amount"
    ).value =
        "8500";


    document.getElementById(
        "distance_from_home"
    ).value =
        "420";


    document.getElementById(
        "account_age_days"
    ).value =
        "120";


    document.getElementById(
        "failed_attempts"
    ).value =
        "3";


    document.getElementById(
        "merchant_category"
    ).value =
        "electronics";


    document.getElementById(
        "payment_method"
    ).value =
        "credit_card";


    document.getElementById(
        "device_changed"
    ).checked =
        true;


    document.getElementById(
        "international"
    ).checked =
        true;


    formMessage.textContent =
        "";
}


/* =====================================
   EVENTS
===================================== */

form.addEventListener(
    "submit",
    analyzeTransaction
);


resetBtn.addEventListener(
    "click",
    resetForm
);


loadDemoBtn.addEventListener(
    "click",
    loadDemo
);


/* =====================================
   START API CHECK
===================================== */

checkApi();


/* Check API every 15 seconds */

setInterval(
    checkApi,
    15000
);