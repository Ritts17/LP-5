#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// Function to print the array
void printArray(const vector<int>& arr) {
    for (int i = 0; i < arr.size(); ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// ----------- Bubble Sort (Serial) ----------------
void bubbleSortSerial(vector<int> &arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i)
        for (int j = 0; j < n - i - 1; ++j)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
}

// ----------- Bubble Sort (Parallel) ----------------
void bubbleSortParallel(vector<int> &arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// ----------- Merge Function ----------------
void merge(vector<int> &arr, int l, int m, int r) {
    vector<int> left(arr.begin() + l, arr.begin() + m + 1);
    vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);

    int i = 0, j = 0, k = l;
    while (i < left.size() && j < right.size())
        arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];

    while (i < left.size()) arr[k++] = left[i++];
    while (j < right.size()) arr[k++] = right[j++];
}

// ----------- Merge Sort (Serial) ----------------
void mergeSortSerial(vector<int> &arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSortSerial(arr, l, m);
        mergeSortSerial(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// ----------- Merge Sort (Parallel) ----------------
void mergeSortParallel(vector<int> &arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortParallel(arr, l, m);

            #pragma omp section
            mergeSortParallel(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

// ----------- Main Function ----------------
int main() {
    int n;
    cout << "Enter size of array: ";
    cin >> n;

    vector<int> arr(n);
    srand(time(0));

    for (int i = 0; i < n; ++i)
        arr[i] = rand() % 10000;

    vector<int> arr1 = arr;
    vector<int> arr2 = arr;
    vector<int> arr3 = arr;
    vector<int> arr4 = arr;

    double start, end;

    // Serial Bubble Sort
    start = omp_get_wtime();
    bubbleSortSerial(arr1);
    end = omp_get_wtime();
    cout << "Serial Bubble Sort Time: " << end - start << " seconds\n";
    cout << "Sorted Array (Serial Bubble Sort): ";
    printArray(arr1);

    // Parallel Bubble Sort
    start = omp_get_wtime();
    bubbleSortParallel(arr2);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << end - start << " seconds\n";
    cout << "Sorted Array (Parallel Bubble Sort): ";
    printArray(arr2);

    // Serial Merge Sort
    start = omp_get_wtime();
    mergeSortSerial(arr3, 0, n - 1);
    end = omp_get_wtime();
    cout << "Serial Merge Sort Time: " << end - start << " seconds\n";
    cout << "Sorted Array (Serial Merge Sort): ";
    printArray(arr3);

    // Parallel Merge Sort
    start = omp_get_wtime();
    mergeSortParallel(arr4, 0, n - 1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << end - start << " seconds\n";
    cout << "Sorted Array (Parallel Merge Sort): ";
    printArray(arr4);

    return 0;
}
