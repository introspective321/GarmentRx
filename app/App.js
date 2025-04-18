import React, { useState } from 'react';
import { StyleSheet, View, Button, Image, Text, FlatList, Alert, ActivityIndicator, ScrollView, SafeAreaView, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import Slider from '@react-native-community/slider';

export default function App() {
    const [segmentedImage, setSegmentedImage] = useState(null);
    const [matches, setMatches] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [originalImage, setOriginalImage] = useState(null);

    const resetResults = () => {
        setSegmentedImage(null);
        setMatches([]);
        setOriginalImage(null);
    };

    const selectImage = async () => {
        const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (!permission.granted) {
            Alert.alert('Permission required', 'Allow access to photos');
            return;
        }

        // Different options for web vs native
        const pickerOptions = Platform.OS === 'web' 
            ? {
                    mediaTypes: ImagePicker.MediaTypeOptions.Images,
                    quality: 1,
                }
            : {
                    mediaTypes: [ImagePicker.MediaType.IMAGE],
                    aspect: [3, 4],
                    quality: 1,
                };

        const result = await ImagePicker.launchImageLibraryAsync(pickerOptions);

        if (!result.canceled) {
            setOriginalImage(result.assets[0].uri);
            uploadImage(result.assets[0].uri);
        }
    };

    const pickImage = async () => {
        await selectImage();
    };

    const uploadImage = async (uri) => {
        setUploading(true);
        try {
            console.log('Preparing to upload image');
            const formData = new FormData();
            
            // Different handling for web platform
            if (Platform.OS === 'web') {
                const response = await fetch(uri);
                const blob = await response.blob();
                formData.append('image', blob, 'image.jpg');
            } else {
                formData.append('image', {
                    uri,
                    name: 'image.jpg',
                    type: 'image/jpeg',
                });
            }

            // Use localhost for web and the IP for native
            const apiUrl = Platform.OS === 'web' 
                ? 'http://localhost:3000/upload' 
                : 'http://172.31.95.11:3000/upload';
            
            console.log(`Sending request to ${apiUrl}`);
            
            const response = await axios.post(apiUrl, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            
            console.log('Response:', response.data);
            setSegmentedImage(response.data.segmented);
            setMatches(response.data.matches);
        } catch (error) {
            console.error('Error details:', error);
            Alert.alert('Upload Failed', 
                `Unable to process image: ${error.message}. Check network and try again.`);
        }
        setUploading(false);
    };

    const renderMatch = ({ item }) => (
        <View style={styles.match}>
            <Image
                source={{ uri: `${Platform.OS === 'web' ? 'http://localhost:3000' : 'http://172.31.95.11:3000'}${item.image}` }}
                style={styles.matchImage}
            />
            <Text style={styles.matchText}>{`${item.color} ${item.style} dress`}</Text>
            <Text style={styles.similarityText}>{`Similarity: ${item.similarity.toFixed(2)}`}</Text>
        </View>
    );

    return (
        <SafeAreaView style={styles.safeArea}>
            <ScrollView contentContainerStyle={styles.scrollContainer}>
                <View style={styles.container}>
                    <Text style={styles.title}>Relove Cloth Matcher - By Anushk</Text>

                    {!uploading && !segmentedImage && (
                        <Button title="Upload Photo" onPress={pickImage} color="#1E90FF" />
                    )}

                    {uploading && (
                        <View style={styles.loadingContainer}>
                            <ActivityIndicator size="large" color="#1E90FF" />
                            <Text style={styles.loadingText}>Processing your image...</Text>
                        </View>
                    )}

                    {originalImage && !uploading && (
                        <View style={styles.resultContainer}>
                            <View style={styles.imageComparison}>
                                {originalImage && (
                                    <View style={styles.imageCol}>
                                        <Text style={styles.imageLabel}>Your Photo</Text>
                                        <Image source={{ uri: originalImage }} style={styles.originalImage} />
                                    </View>
                                )}
                                {segmentedImage && (
                                    <View style={styles.imageCol}>
                                        <Text style={styles.imageLabel}>Detected Item</Text>
                                        <Image
                                            source={{ uri: `${Platform.OS === 'web' ? 'http://localhost:3000' : 'http://172.31.95.11:3000'}${segmentedImage}` }}
                                            style={styles.segmentedImage}
                                        />
                                    </View>
                                )}
                            </View>

                            <View style={styles.buttonContainer}>
                                <Button
                                    title="Try Another Photo"
                                    onPress={() => {
                                        resetResults();
                                        pickImage();
                                    }}
                                    color="#1E90FF"
                                />
                                <View style={styles.buttonSpacer} />
                                <Button title="Start Over" onPress={resetResults} color="#FF6347" />
                            </View>

                            {matches.length > 0 && (
                                <View style={styles.matchesContainer}>
                                    <Text style={styles.sectionTitle}>Similar Items</Text>
                                </View>
                            )}
                        </View>
                    )}
                </View>
            </ScrollView>

            {matches.length > 0 && originalImage && !uploading && (
                <FlatList
                    data={matches}
                    renderItem={renderMatch}
                    keyExtractor={(item) => item.id}
                    style={styles.matchList}
                    numColumns={2}
                    columnWrapperStyle={styles.matchRow}
                    ListHeaderComponent={<View style={{ height: 10 }} />}
                />
            )}
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safeArea: {
        flex: 1,
        backgroundColor: '#F5F5F5',
    },
    scrollContainer: {
        flexGrow: 1,
        paddingBottom: 20,
    },
    container: {
        flex: 1,
        alignItems: 'center',
        paddingHorizontal: 15,
    },
    title: {
        fontSize: 28,
        fontWeight: '700',
        color: '#1E90FF',
        marginVertical: 15,
        textAlign: 'center',
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    loadingText: {
        marginTop: 10,
        fontSize: 18,
        color: '#333',
    },
    resultContainer: {
        width: '100%',
        alignItems: 'center',
        backgroundColor: '#FFF',
        borderRadius: 12,
        padding: 15,
        elevation: 3,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
    },
    imageComparison: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        width: '100%',
        marginBottom: 20,
    },
    imageCol: {
        alignItems: 'center',
        width: '48%',
    },
    imageLabel: {
        fontSize: 16,
        fontWeight: '600',
        color: '#1E90FF',
        marginBottom: 5,
    },
    originalImage: {
        width: 180,
        height: 240,
        resizeMode: 'contain',
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#EEE',
    },
    segmentedImage: {
        width: 180,
        height: 240,
        resizeMode: 'contain',
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#EEE',
    },
    matchesContainer: {
        width: '100%',
        marginTop: 15,
        marginBottom: 10,
    },
    sectionTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: '#1E90FF',
        marginBottom: 10,
    },
    matchList: {
        width: '100%',
        paddingHorizontal: 15,
        marginBottom: 20,
    },
    matchRow: {
        justifyContent: 'space-around',
    },
    match: {
        flex: 0.45,
        alignItems: 'center',
        margin: 8,
        padding: 10,
        backgroundColor: '#F9F9F9',
        borderRadius: 10,
        elevation: 3,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 3,
    },
    matchImage: {
        width: 120,
        height: 160,
        resizeMode: 'contain',
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#DDD',
    },
    matchText: {
        fontSize: 14,
        fontWeight: '500',
        color: '#333',
        marginTop: 5,
        textAlign: 'center',
    },
    similarityText: {
        fontSize: 12,
        color: '#666',
        marginTop: 2,
    },
    buttonContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        marginTop: 20,
        marginBottom: 10,
    },
    buttonSpacer: {
        width: 15,
    },
});
